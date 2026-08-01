//! FFN block fusion pass.
//!
//! Detects the pattern:
//! ```text
//! x ──→ LayerNorm ──→ Linear(up) ──→ GELU ──→ Linear(down) ──→ Add(residual)
//! ```
//!
//! And replaces it with a single [`FusedFfnBlock`](crate::op::Op::FusedFfnBlock).

use crate::error::Result;
use crate::graph::{Graph, NodeId};
use crate::op::{Activation, Op};
use crate::pass::TransformPass;
use crate::graph::Node;

/// Fuses feed-forward network sub-blocks in transformer layers.
pub struct FfnBlockFusion;

impl TransformPass for FfnBlockFusion {
    fn name(&self) -> &'static str {
        "ffn_block_fusion"
    }

    fn transform(&self, graph: &mut Graph) -> Result<usize> {
        let mut fused_count = 0;

        loop {
            let candidate = find_ffn_block_entry(graph);
            let Some(params) = candidate else {
                break;
            };

            fuse_ffn_block(graph, &params)?;
            fused_count += 1;
        }

        if fused_count > 0 {
            tracing::info!("ffn_block_fusion: fused {fused_count} FFN block(s)");
        }

        Ok(fused_count)
    }
}

struct FfnBlockParams {
    nodes_to_remove: Vec<NodeId>,
    ln_input_tensor: crate::graph::TensorId,
    residual_tensor: crate::graph::TensorId,
    output_tensor: crate::graph::TensorId,
    hidden_dim: usize,
    intermediate_dim: usize,
    activation: Activation,
    has_bias: bool,
}

fn find_ffn_block_entry(graph: &Graph) -> Option<FfnBlockParams> {
    let order = graph.topological_order();

    for &nid in &order {
        let node = graph.node_by_id(nid)?;

        // Step 1: Must be a LayerNorm.
        if !matches!(node.op(), Op::LayerNorm { .. }) {
            continue;
        }

        // Step 2: Successor must be a Linear (up projection), NOT a QKV (3× hidden_dim).
        let succs = graph.successors(nid);
        if succs.len() != 1 {
            continue;
        }
        let up_id = succs[0];
        let up_node = graph.node_by_id(up_id)?;
        let (intermediate_dim, has_bias) = match up_node.op() {
            Op::Linear { out_features, bias } => (*out_features, *bias),
            _ => continue,
        };

        let ln_input = *node.inputs().first()?;
        let ln_input_meta = graph.tensor(ln_input)?;
        let hidden_dim = *ln_input_meta.shape().last()?;

        // Skip if this looks like a QKV projection (3× multiplier).
        if intermediate_dim == 3 * hidden_dim {
            continue;
        }

        // Step 3: Successor of up-proj must be an activation (GELU/ReLU/SiLU).
        let up_succs = graph.successors(up_id);
        if up_succs.len() != 1 {
            continue;
        }
        let act_id = up_succs[0];
        let act_node = graph.node_by_id(act_id)?;
        let activation = match act_node.op() {
            Op::Gelu => Activation::Gelu,
            Op::Relu => Activation::Relu,
            Op::Silu => Activation::Silu,
            _ => continue,
        };

        // Step 4: Successor of activation must be Linear (down projection).
        let act_succs = graph.successors(act_id);
        if act_succs.len() != 1 {
            continue;
        }
        let down_id = act_succs[0];
        let down_node = graph.node_by_id(down_id)?;
        match down_node.op() {
            Op::Linear { out_features, .. } if *out_features == hidden_dim => {}
            _ => continue,
        }

        // Step 5: Successor of down-proj must be Add (residual).
        let down_succs = graph.successors(down_id);
        if down_succs.len() != 1 {
            continue;
        }
        let add_id = down_succs[0];
        let add_node = graph.node_by_id(add_id)?;
        if !matches!(add_node.op(), Op::Add) {
            continue;
        }

        let output_tensor = *add_node.outputs().first()?;
        let down_output = *down_node.outputs().first()?;
        let residual_tensor = add_node
            .inputs()
            .iter()
            .copied()
            .find(|tid| *tid != down_output)
            .unwrap_or(ln_input);

        return Some(FfnBlockParams {
            nodes_to_remove: vec![nid, up_id, act_id, down_id, add_id],
            ln_input_tensor: ln_input,
            residual_tensor,
            output_tensor,
            hidden_dim,
            intermediate_dim,
            activation,
            has_bias,
        });
    }

    None
}

fn fuse_ffn_block(graph: &mut Graph, params: &FfnBlockParams) -> Result<()> {
    let output_meta = graph
        .tensor(params.output_tensor)
        .ok_or_else(|| crate::error::Error::FusionFailed {
            pattern_name: "ffn_block".into(),
            reason: "output tensor not found".into(),
        })?;
    let _dtype = output_meta.dtype();

    // To preserve the output tensor when removing nodes, we temporarily detach
    // the node that produces the output tensor from the node's output list.
    for &nid in &params.nodes_to_remove {
        if let Some(node) = graph.nodes.get_mut(&nid) {
            node.outputs.retain(|&id| id != params.output_tensor);
        }
        graph.remove_node(nid);
    }

    let fused_id = graph.alloc_node_id();

    // Update the original output tensor to have the fused node as its producer
    graph.update_tensor_producer(params.output_tensor, Some(fused_id));

    let mut inputs = vec![params.ln_input_tensor];
    if params.residual_tensor != params.ln_input_tensor {
        inputs.push(params.residual_tensor);
    }

    let fused_node = Node::new(
        fused_id,
        "fused_ffn_block",
        Op::FusedFfnBlock {
            hidden_dim: params.hidden_dim,
            intermediate_dim: params.intermediate_dim,
            activation: params.activation,
            has_bias: params.has_bias,
        },
        inputs,
        vec![params.output_tensor],
    );

    graph.insert_node(fused_node);
    Ok(())
}
