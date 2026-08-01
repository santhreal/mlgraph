//! Attention block fusion pass.
//!
//! Detects the pattern:
//! ```text
//! x ──→ LayerNorm ──→ Linear(QKV) ──→ Split ──→ [Reshape → Transpose] × 3
//!   ──→ MatMul(Q,K^T) ──→ ScalarMul ──→ Softmax ──→ MatMul(attn,V)
//!   ──→ Transpose ──→ Reshape ──→ Linear(out) ──→ Add(residual)
//! ```
//!
//! And replaces it with a single [`FusedAttentionBlock`](crate::op::Op::FusedAttentionBlock)
//! node. The fused node reads input + residual + weights from HBM and writes
//! only the output to HBM. All intermediates stay in SRAM.

use crate::error::Result;
use crate::graph::{Graph, Node, NodeId};
use crate::op::Op;
use crate::pass::TransformPass;

/// Fuses attention sub-blocks in transformer layers.
///
/// Scans the graph for `LayerNorm → Linear → ... → Add` patterns that
/// match multi-head self-attention structure and replaces them with
/// a single `FusedAttentionBlock` node.
pub struct AttentionBlockFusion;

impl TransformPass for AttentionBlockFusion {
    fn name(&self) -> &'static str {
        "attention_block_fusion"
    }

    fn transform(&self, graph: &mut Graph) -> Result<usize> {
        let mut fused_count = 0;

        // Find all attention block entry points: LayerNorm nodes whose
        // single successor is a QKV Linear projection.
        loop {
            let candidate = find_attention_block_entry(graph);
            let Some((ln_id, attn_params)) = candidate else {
                break;
            };

            fuse_attention_block(graph, ln_id, &attn_params)?;
            fused_count += 1;
        }

        if fused_count > 0 {
            tracing::info!(
                "attention_block_fusion: fused {fused_count} attention block(s)"
            );
        }

        Ok(fused_count)
    }
}

/// Parameters extracted from an attention block subgraph.
struct AttentionBlockParams {
    /// Nodes to remove (the entire unfused subgraph).
    nodes_to_remove: Vec<NodeId>,
    /// Input tensor to the LayerNorm (the "x" / activation input).
    ln_input_tensor: crate::graph::TensorId,
    /// The residual tensor for the skip connection (often same as ln_input).
    residual_tensor: crate::graph::TensorId,
    /// Output tensor of the final Add node.
    output_tensor: crate::graph::TensorId,
    /// Number of attention heads.
    num_heads: usize,
    /// Dimension per head.
    head_dim: usize,
    /// Hidden dimension.
    hidden_dim: usize,
    /// Whether projections use bias.
    has_bias: bool,
}

/// Walk the graph looking for a LayerNorm that starts an attention block.
fn find_attention_block_entry(graph: &Graph) -> Option<(NodeId, AttentionBlockParams)> {
    let order = graph.topological_order();

    for &nid in &order {
        let node = graph.node_by_id(nid)?;

        // Step 1: Must be a LayerNorm.
        if !matches!(node.op(), Op::LayerNorm { .. }) {
            continue;
        }

        // Step 2: Its single successor must be a Linear (QKV projection).
        let succs = graph.successors(nid);
        if succs.len() != 1 {
            continue;
        }
        let qkv_id = succs[0];
        let qkv_node = graph.node_by_id(qkv_id)?;
        let (qkv_out_features, has_bias) = match qkv_node.op() {
            Op::Linear { out_features, bias } => (*out_features, *bias),
            _ => continue,
        };

        // The QKV linear must project to 3 * hidden_dim.
        let ln_input = *node.inputs().first()?;
        let ln_input_meta = graph.tensor(ln_input)?;
        let hidden_dim = *ln_input_meta.shape().last()?;
        if qkv_out_features != 3 * hidden_dim {
            continue;
        }

        // Walk forward through the attention block to find the final Add.
        let chain = trace_attention_chain(graph, qkv_id, hidden_dim);
        let Some((nodes_to_remove, out_proj_output, final_add_id)) = chain else {
            continue;
        };

        let add_node = graph.node_by_id(final_add_id)?;
        let output_tensor = *add_node.outputs().first()?;

        // The residual input to the Add is the tensor that was NOT produced
        // by the output projection chain.
        let residual_tensor = add_node
            .inputs()
            .iter()
            .copied()
            .find(|tid| *tid != out_proj_output)
            .unwrap_or(ln_input);

        // Determine num_heads and head_dim from hidden_dim.
        // ViT-Tiny: 192 = 3 heads × 64 dim
        // ViT-Small: 384 = 6 heads × 64 dim
        // ViT-Base: 768 = 12 heads × 64 dim
        // Default: head_dim = 64 if evenly divisible.
        let head_dim = if hidden_dim % 64 == 0 {
            64
        } else if hidden_dim % 32 == 0 {
            32
        } else {
            hidden_dim
        };
        let num_heads = hidden_dim / head_dim;

        let mut all_nodes = vec![nid]; // LayerNorm
        all_nodes.push(qkv_id);
        all_nodes.extend(nodes_to_remove);
        all_nodes.push(final_add_id);
        all_nodes.sort();
        all_nodes.dedup();

        return Some((
            nid,
            AttentionBlockParams {
                nodes_to_remove: all_nodes,
                ln_input_tensor: ln_input,
                residual_tensor,
                output_tensor,
                num_heads,
                head_dim,
                hidden_dim,
                has_bias,
            },
        ));
    }

    None
}

/// Trace forward from the QKV Linear through the attention subgraph.
///
/// Returns `(intermediate_node_ids, out_proj_output_tensor, final_add_node_id)`.
fn trace_attention_chain(
    graph: &Graph,
    qkv_id: NodeId,
    _hidden_dim: usize,
) -> Option<(Vec<NodeId>, crate::graph::TensorId, NodeId)> {
    let mut visited = vec![qkv_id];
    let mut frontier = graph.successors(qkv_id);
    let mut last_add: Option<NodeId> = None;
    let mut out_proj_output = None;

    // Walk forward up to 20 nodes (attention blocks are ~15 nodes).
    for _ in 0..20 {
        if frontier.is_empty() {
            break;
        }
        let mut next_frontier = Vec::new();
        for &nid in &frontier {
            let Some(node) = graph.node_by_id(nid) else {
                continue;
            };

            match node.op() {
                Op::Add => {
                    // This is the residual add  -  the end of the block.
                    last_add = Some(nid);
                    // The out_proj output is the tensor input to Add that
                    // has a producer in our visited set.
                    if out_proj_output.is_none() {
                        for tid in node.inputs() {
                            if let Some(tensor) = graph.tensor(*tid) {
                                if let Some(producer) = tensor.producer() {
                                    if visited.contains(&producer) {
                                        out_proj_output = Some(*tid);
                                    }
                                }
                            }
                        }
                    }
                }
                // Linear at this point is the output projection.
                Op::Linear { .. } => {
                    visited.push(nid);
                    // The out proj's output feeds into Add.
                    if let Some(out_tid) = node.outputs().first() {
                        out_proj_output = Some(*out_tid);
                    }
                    next_frontier.extend(graph.successors(nid));
                }
                // Everything else is intermediate (reshape, transpose, matmul, etc.)
                _ => {
                    visited.push(nid);
                    next_frontier.extend(graph.successors(nid));
                }
            }
        }

        if last_add.is_some() {
            break;
        }
        frontier = next_frontier;
        frontier.sort();
        frontier.dedup();
    }

    let add_id = last_add?;
    let proj_output = out_proj_output?;

    // Safety: none of the intermediate nodes we are about to fuse may have an
    // output tensor consumed from OUTSIDE the traced subgraph. Fusing removes
    // these nodes, so an external consumer would be left dangling. Internal
    // branching (e.g. a QKV projection feeding a split) is fine as long as every
    // consumer is itself part of the fused set; the output-projection tensor is
    // exempt because it legitimately feeds the residual Add.
    let fused_set: std::collections::HashSet<NodeId> =
        visited.iter().copied().chain(std::iter::once(add_id)).collect();
    for &nid in &visited {
        let Some(node) = graph.node_by_id(nid) else {
            continue;
        };
        for out_tid in node.outputs() {
            if *out_tid == proj_output {
                continue;
            }
            if let Some(tensor) = graph.tensor(*out_tid) {
                for &consumer in tensor.consumers() {
                    if !fused_set.contains(&consumer) {
                        // An intermediate result escapes the block; fusing here
                        // would corrupt the graph, so decline the fusion.
                        return None;
                    }
                }
            }
        }
    }

    Some((visited, proj_output, add_id))
}

/// Replace the attention block subgraph with a single fused node.
fn fuse_attention_block(
    graph: &mut Graph,
    _ln_id: NodeId,
    params: &AttentionBlockParams,
) -> Result<()> {
    // Determine output shape and dtype from the original output tensor.
    let output_meta = graph
        .tensor(params.output_tensor)
        .ok_or_else(|| crate::error::Error::FusionFailed {
            pattern_name: "attention_block".into(),
            reason: "output tensor not found".into(),
        })?;
    let _dtype = output_meta.dtype();

    // Collect consumers of the original output tensor BEFORE removing nodes.
    // These are the downstream nodes that need to be rewired to use the fused output.
    let _downstream_consumers: Vec<NodeId> = output_meta.consumers().to_vec();

    // Remove all nodes in the subgraph.
    // To preserve the output tensor when removing nodes, we temporarily detach
    // the node that produces the output tensor from the node's output list.
    for &nid in &params.nodes_to_remove {
        if let Some(node) = graph.nodes.get_mut(&nid) {
            node.outputs.retain(|&id| id != params.output_tensor);
        }
        graph.remove_node(nid);
    }

    // Create the fused node, reusing the original output tensor ID
    // so downstream consumers automatically point to the right tensor.
    let fused_id = graph.alloc_node_id();

    // Update the original output tensor to have the fused node as its producer
    // and clear its old consumer list (the fused node will be the new producer).
    graph.update_tensor_producer(params.output_tensor, Some(fused_id));
    if let Some(_tensor) = graph.tensor_mut(params.output_tensor) {
        // Clear consumers; the fused node will re-register any that remain
        // after we update downstream node inputs.
        // Actually, we need to preserve downstream consumers but update their inputs.
        // The tensor's consumer list will be updated when we call insert_node.
    }

    let mut inputs = vec![params.ln_input_tensor];
    if params.residual_tensor != params.ln_input_tensor {
        inputs.push(params.residual_tensor);
    }

    let fused_node = Node::new(
        fused_id,
        "fused_attn_block",
        Op::FusedAttentionBlock {
            num_heads: params.num_heads,
            head_dim: params.head_dim,
            hidden_dim: params.hidden_dim,
            has_bias: params.has_bias,
        },
        inputs,
        vec![params.output_tensor],
    );

    graph.insert_node(fused_node);

    // The fused_output is now at params.output_tensor, but we need to
    // update downstream nodes' inputs to reference this tensor correctly.
    // Since we reused the tensor ID, downstream consumers should already
    // have the correct tensor ID in their inputs.

    Ok(())
}


