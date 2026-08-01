//! Integration tests for mlgraph.
//!
//! Tests cover:
//! - Fusion correctness (small graph → fuse → verify invariants)
//! - Shape inference adversarial tests (bad dims, bad perms)
//! - Bandwidth analysis tests

use mlgraph::analysis::bandwidth::BandwidthAnalysis;
use mlgraph::fusion::attn_block::AttentionBlockFusion;
use mlgraph::fusion::ffn_block::FfnBlockFusion;
use mlgraph::op::Op;
use mlgraph::pass::{AnalysisPass, TransformPass};
use mlgraph::{DType, Graph};

// =============================================================================
// Fusion Correctness Tests
// =============================================================================

/// Build a minimal attention block graph for testing.
/// Pattern: x → LayerNorm → Linear(QKV) → ... → Add(residual)
fn build_test_attention_block() -> (Graph, mlgraph::graph::TensorId) {
    let mut graph = Graph::new("test_attn");

    // Input: [batch=1, seq=4, hidden=192] like ViT-Tiny
    let x = graph.input("x", &[1, 4, 192], DType::F16);

    // LayerNorm
    let ln_out = graph.node("ln", Op::LayerNorm { eps: 1e-6 }, &[x]).unwrap();

    // QKV Linear: 192 → 576 (3 * 192)
    let qkv_out = graph
        .node("qkv", Op::Linear { out_features: 576, bias: true }, &[ln_out[0]])
        .unwrap();

    // Split into Q, K, V (3 parts of 192 each)
    let split_out = graph
        .node(
            "split",
            Op::Split {
                dim: 2,
                sections: vec![192, 192, 192],
            },
            &[qkv_out[0]],
        )
        .unwrap();

    // Simplified: instead of full attention math, we'll just add the outputs
    // to simulate the output projection + residual pattern
    let add_out = graph.node("add", Op::Add, &[split_out[0], x]).unwrap();

    graph.mark_output(add_out[0]);

    (graph, add_out[0])
}

#[test]
fn test_attention_block_fusion_rewires_consumers() {
    // Build a graph with an attention block followed by another operation
    let mut graph = Graph::new("test_attn_chain");

    // Input
    let x = graph.input("x", &[1, 4, 192], DType::F16);

    // First: a simple chain that won't be fused
    let ln1_out = graph.node("ln1", Op::LayerNorm { eps: 1e-6 }, &[x]).unwrap();
    let proj_out = graph
        .node("proj", Op::Linear { out_features: 192, bias: true }, &[ln1_out[0]])
        .unwrap();
    let add_out = graph.node("add", Op::Add, &[proj_out[0], x]).unwrap();

    // Followed by another operation that consumes the output
    let ln2_out = graph.node("ln2", Op::LayerNorm { eps: 1e-6 }, &[add_out[0]]).unwrap();
    graph.mark_output(ln2_out[0]);

    // Store the output tensor ID
    let original_output = add_out[0];

    // Apply fusion pass
    let pass = AttentionBlockFusion;
    let result = pass.transform(&mut graph);

    // The graph should be valid after fusion (no panic, no error)
    assert!(result.is_ok());

    // Verify the graph still has a valid structure
    // The output tensor should still exist
    assert!(graph.tensor(original_output).is_some());

    // The ln2 node should still have a valid input
    let ln2_node = graph
        .nodes()
        .find(|n| n.name() == "ln2")
        .expect("ln2 node should exist");
    assert_eq!(ln2_node.inputs().len(), 1);
}

#[test]
fn test_ffn_block_fusion_with_gelu() {
    let mut graph = Graph::new("test_ffn_gelu");

    // Input
    let x = graph.input("x", &[1, 4, 192], DType::F16);

    // FFN pattern: LayerNorm → Linear(up) → Gelu → Linear(down) → Add
    let ln_out = graph.node("ln", Op::LayerNorm { eps: 1e-6 }, &[x]).unwrap();
    let up_out = graph
        .node("up", Op::Linear { out_features: 768, bias: true }, &[ln_out[0]])
        .unwrap();
    let gelu_out = graph.node("gelu", Op::Gelu, &[up_out[0]]).unwrap();
    let down_out = graph
        .node("down", Op::Linear { out_features: 192, bias: true }, &[gelu_out[0]])
        .unwrap();
    let add_out = graph.node("add", Op::Add, &[down_out[0], x]).unwrap();

    // Followed by another operation
    let ln2_out = graph.node("ln2", Op::LayerNorm { eps: 1e-6 }, &[add_out[0]]).unwrap();
    graph.mark_output(ln2_out[0]);

    let original_output = add_out[0];

    // Apply fusion
    let pass = FfnBlockFusion;
    let result = pass.transform(&mut graph);

    assert!(result.is_ok());
    assert!(graph.tensor(original_output).is_some());

    // The ln2 node should still consume the fused output
    let ln2_node = graph.nodes().find(|n| n.name() == "ln2").unwrap();
    assert_eq!(ln2_node.inputs().len(), 1);

    // The fused node must be registered as a consumer of every one of its input
    // tensors, or successor()/topological_order() would silently miss the edge.
    let fused = graph
        .nodes()
        .find(|n| matches!(n.op(), Op::FusedFfnBlock { .. }))
        .expect("a fused FFN node should exist after fusion");
    let fused_id = fused.id();
    let fused_inputs = fused.inputs().to_vec();
    assert!(!fused_inputs.is_empty(), "fused node must have inputs");
    for tid in fused_inputs {
        let consumers = graph.tensor(tid).expect("fused input tensor").consumers();
        assert!(
            consumers.contains(&fused_id),
            "input tensor {tid:?} must list the fused node {fused_id:?} as a consumer"
        );
    }
}

#[test]
fn test_ffn_block_fusion_with_relu() {
    let mut graph = Graph::new("test_ffn_relu");

    let x = graph.input("x", &[1, 4, 192], DType::F16);

    // FFN with ReLU
    let ln_out = graph.node("ln", Op::LayerNorm { eps: 1e-6 }, &[x]).unwrap();
    let up_out = graph
        .node("up", Op::Linear { out_features: 768, bias: true }, &[ln_out[0]])
        .unwrap();
    let relu_out = graph.node("relu", Op::Relu, &[up_out[0]]).unwrap();
    let down_out = graph
        .node("down", Op::Linear { out_features: 192, bias: true }, &[relu_out[0]])
        .unwrap();
    let add_out = graph.node("add", Op::Add, &[down_out[0], x]).unwrap();
    graph.mark_output(add_out[0]);

    let pass = FfnBlockFusion;
    let result = pass.transform(&mut graph);

    // Should find and fuse one block
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 1);
}

#[test]
fn test_ffn_block_fusion_with_silu() {
    let mut graph = Graph::new("test_ffn_silu");

    let x = graph.input("x", &[1, 4, 192], DType::F16);

    // FFN with SiLU (this was the missing activation)
    let ln_out = graph.node("ln", Op::LayerNorm { eps: 1e-6 }, &[x]).unwrap();
    let up_out = graph
        .node("up", Op::Linear { out_features: 768, bias: true }, &[ln_out[0]])
        .unwrap();
    let silu_out = graph.node("silu", Op::Silu, &[up_out[0]]).unwrap();
    let down_out = graph
        .node("down", Op::Linear { out_features: 192, bias: true }, &[silu_out[0]])
        .unwrap();
    let add_out = graph.node("add", Op::Add, &[down_out[0], x]).unwrap();
    graph.mark_output(add_out[0]);

    let pass = FfnBlockFusion;
    let result = pass.transform(&mut graph);

    // Should find and fuse one block (SiLU is now supported)
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 1);
}

// =============================================================================
// Shape Inference Adversarial Tests
// =============================================================================

#[test]
fn test_matmul_valid_shapes() {
    let op = Op::MatMul;

    // Valid: [M, K] @ [K, N] = [M, N]
    let result = op.infer_shapes(&[&[4, 8], &[8, 16]]);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), vec![vec![4, 16]]);

    // Valid with batch: [B, M, K] @ [B, K, N] = [B, M, N]
    let result = op.infer_shapes(&[&[2, 4, 8], &[2, 8, 16]]);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), vec![vec![2, 4, 16]]);
}

#[test]
fn test_matmul_invalid_contraction_dimension() {
    let op = Op::MatMul;

    // Invalid: [M, K1] @ [K2, N] where K1 != K2
    let result = op.infer_shapes(&[&[4, 8], &[10, 16]]);
    assert!(result.is_err());

    let err = result.unwrap_err().to_string();
    assert!(err.contains("contraction dimension mismatch"));
    assert!(err.contains("a[-1]=8"));
    assert!(err.contains("b[-2]=10"));
}

#[test]
fn test_matmul_not_enough_inputs() {
    let op = Op::MatMul;

    let result = op.infer_shapes(&[&[4, 8]]);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("expected 2 inputs"));
}

#[test]
fn test_matmul_not_enough_dimensions() {
    let op = Op::MatMul;

    // 1D inputs not allowed
    let result = op.infer_shapes(&[&[8], &[8]]);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("at least 2-dimensional"));
}

#[test]
fn test_transpose_valid_permutation() {
    let op = Op::Transpose {
        perm: vec![1, 0, 2],
    };

    // Valid: [B, S, H] → [S, B, H]
    let result = op.infer_shapes(&[&[2, 4, 8]]);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), vec![vec![4, 2, 8]]);
}

#[test]
fn test_transpose_out_of_bounds_permutation() {
    // Permutation index >= rank
    let op = Op::Transpose { perm: vec![1, 3, 0] };

    let result = op.infer_shapes(&[&[2, 4, 8]]);
    assert!(result.is_err());

    let err = result.unwrap_err().to_string();
    assert!(err.contains("out of bounds"));
}

#[test]
fn test_transpose_duplicate_permutation_indices() {
    // Duplicate indices are invalid
    let op = Op::Transpose { perm: vec![0, 1, 1] };

    let result = op.infer_shapes(&[&[2, 4, 8]]);
    assert!(result.is_err());

    let err = result.unwrap_err().to_string();
    assert!(err.contains("appears multiple times"));
    assert!(err.contains("must be unique"));
}

#[test]
fn test_transpose_permutation_length_mismatch() {
    let op = Op::Transpose {
        perm: vec![0, 1, 2, 3],
    };

    // Input is rank 3 but perm is length 4
    let result = op.infer_shapes(&[&[2, 4, 8]]);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("permutation length"));
}

#[test]
fn test_transpose_empty_input() {
    let op = Op::Transpose {
        perm: vec![0, 1],
    };

    let result = op.infer_shapes(&[]);
    assert!(result.is_err());
}

// =============================================================================
// Bandwidth Analysis Tests
// =============================================================================

#[test]
fn test_bandwidth_analysis_empty_graph() {
    let graph = Graph::new("empty");

    let analysis = BandwidthAnalysis;
    let result = analysis.analyze(&graph);

    assert!(result.is_ok());
    let report = result.unwrap();
    assert_eq!(report.total_hbm_traffic, 0);
}

#[test]
fn test_bandwidth_analysis_simple_graph() {
    let mut graph = Graph::new("simple");

    // Simple: x → Linear → output
    let x = graph.input("x", &[1, 4, 192], DType::F16);
    let out = graph
        .node("linear", Op::Linear { out_features: 768, bias: true }, &[x])
        .unwrap();
    graph.mark_output(out[0]);

    let analysis = BandwidthAnalysis;
    let result = analysis.analyze(&graph);

    assert!(result.is_ok());
    let report = result.unwrap();
    // Should have non-zero traffic for input + weights + bias + output
    assert!(report.total_hbm_traffic > 0);
}

#[test]
fn test_bandwidth_analysis_with_matmul() {
    let mut graph = Graph::new("matmul_test");

    let a = graph.input("a", &[4, 8], DType::F16);
    let b = graph.input("b", &[8, 16], DType::F16);
    let out = graph.node("matmul", Op::MatMul, &[a, b]).unwrap();
    graph.mark_output(out[0]);

    let analysis = BandwidthAnalysis;
    let result = analysis.analyze(&graph);

    assert!(result.is_ok());
    let report = result.unwrap();
    assert!(report.total_hbm_traffic > 0);
}

#[test]
fn test_bandwidth_analysis_with_activations() {
    let mut graph = Graph::new("activations");

    let x = graph.input("x", &[1, 4, 192], DType::F16);

    // Test various activations
    let gelu_out = graph.node("gelu", Op::Gelu, &[x]).unwrap();
    let relu_out = graph.node("relu", Op::Relu, &[gelu_out[0]]).unwrap();
    let silu_out = graph.node("silu", Op::Silu, &[relu_out[0]]).unwrap();

    graph.mark_output(silu_out[0]);

    let analysis = BandwidthAnalysis;
    let result = analysis.analyze(&graph);

    assert!(result.is_ok());
}

// =============================================================================
// Op variant tests
// =============================================================================

#[test]
fn test_silu_op_properties() {
    let op = Op::Silu;

    assert_eq!(op.name(), "silu");
    assert!(!op.is_fused());
    assert_eq!(op.num_outputs(), 1);

    // Shape inference should preserve shape
    let result = op.infer_shapes(&[&[2, 4, 8]]);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), vec![vec![2, 4, 8]]);
}

#[test]
fn test_fused_ops_report_correct_names() {
    let attn = Op::FusedAttentionBlock {
        num_heads: 12,
        head_dim: 64,
        hidden_dim: 768,
        has_bias: true,
    };
    assert_eq!(attn.name(), "fused_attention_block");
    assert!(attn.is_fused());

    let ffn = Op::FusedFfnBlock {
        hidden_dim: 768,
        intermediate_dim: 3072,
        activation: mlgraph::op::Activation::Gelu,
        has_bias: true,
    };
    assert_eq!(ffn.name(), "fused_ffn_block");
    assert!(ffn.is_fused());
}

// =============================================================================
// Graph structural tests
// =============================================================================

#[test]
fn test_graph_node_removal_updates_tensor_producer() {
    let mut graph = Graph::new("test");

    let x = graph.input("x", &[1, 4, 192], DType::F16);
    let out = graph.node("linear", Op::Linear { out_features: 768, bias: false }, &[x]).unwrap();

    // Before removal, the output tensor has a producer
    let tensor = graph.tensor(out[0]).unwrap();
    assert!(tensor.producer().is_some());

    // Get the node ID
    let node_id = tensor.producer().unwrap();

    // Remove the node
    graph.remove_node(node_id);

    // After removal, the node's outputs are removed from the graph.
    let tensor = graph.tensor(out[0]);
    assert!(tensor.is_none());
}

#[test]
fn test_graph_tensor_producer_update() {
    let mut graph = Graph::new("test");

    let x = graph.input("x", &[1, 4, 192], DType::F16);
    let out = graph.node("ln", Op::LayerNorm { eps: 1e-6 }, &[x]).unwrap();

    let tensor_id = out[0];
    let old_producer = graph.tensor(tensor_id).unwrap().producer().unwrap();

    // Allocate a new node ID (simulating what fusion would do)
    let new_node_id = graph.alloc_node_id();

    // Update the tensor producer
    graph.update_tensor_producer(tensor_id, Some(new_node_id));

    let new_producer = graph.tensor(tensor_id).unwrap().producer().unwrap();
    assert_ne!(old_producer, new_producer);
    assert_eq!(new_producer, new_node_id);
}

#[test]
fn test_graph_topological_ordering() {
    let mut graph = Graph::new("test");

    // Build a small graph with dependencies
    let x = graph.input("x", &[1, 4, 192], DType::F16);
    let a = graph.node("a", Op::Gelu, &[x]).unwrap();
    let b = graph.node("b", Op::Relu, &[a[0]]).unwrap();
    let c = graph.node("c", Op::Gelu, &[b[0]]).unwrap();

    graph.mark_output(c[0]);

    let order = graph.topological_order();

    // Should have 3 nodes
    assert_eq!(order.len(), 3);

    // In topological order, dependencies come before dependents
    // We know the structure: x -> a -> b -> c
    // The exact order might depend on ID allocation, but a must come before b and c
    let a_idx = order.iter().position(|&id| {
        graph.node_by_id(id).map(|n| n.name() == "a").unwrap_or(false)
    }).unwrap();
    let b_idx = order.iter().position(|&id| {
        graph.node_by_id(id).map(|n| n.name() == "b").unwrap_or(false)
    }).unwrap();
    let c_idx = order.iter().position(|&id| {
        graph.node_by_id(id).map(|n| n.name() == "c").unwrap_or(false)
    }).unwrap();

    assert!(a_idx < b_idx, "node 'a' should come before node 'b'");
    assert!(b_idx < c_idx, "node 'b' should come before node 'c'");
}

// =============================================================================
// FLOPs calculation tests
// =============================================================================

#[test]
fn test_flops_matmul() {
    let op = Op::MatMul;

    // [4, 8] @ [8, 16] = [4, 16]
    // FLOPs = 2 * M * K * N = 2 * 4 * 8 * 16 = 1024
    let result = op.flops(&[&[4, 8], &[8, 16]], &[&[4, 16]]);
    assert_eq!(result, 1024);

    // With batch: [2, 4, 8] @ [2, 8, 16] = [2, 4, 16]
    // FLOPs = 2 * batch * M * K * N = 2 * 2 * 4 * 8 * 16 = 2048
    let result = op.flops(&[&[2, 4, 8], &[2, 8, 16]], &[&[2, 4, 16]]);
    assert_eq!(result, 2048);
}

#[test]
fn test_flops_activation_variants() {
    // Gelu: 8 FLOPs per element
    let gelu = Op::Gelu;
    let result = gelu.flops(&[&[4, 8]], &[&[4, 8]]);
    assert_eq!(result, 8 * 32); // 8 * 32 elements

    // Relu: 1 FLOP per element
    let relu = Op::Relu;
    let result = relu.flops(&[&[4, 8]], &[&[4, 8]]);
    assert_eq!(result, 32); // 32 elements

    // Silu: 8 FLOPs per element
    let silu = Op::Silu;
    let result = silu.flops(&[&[4, 8]], &[&[4, 8]]);
    assert_eq!(result, 8 * 32); // 8 * 32 elements
}

#[test]
fn test_flops_fused_ffn_with_different_activations() {
    use mlgraph::op::Activation;

    let base_shape = &[1, 4, 192usize];
    let output_shape = &[1, 4, 192usize];

    // FFN with Gelu
    let ffn_gelu = Op::FusedFfnBlock {
        hidden_dim: 192,
        intermediate_dim: 768,
        activation: Activation::Gelu,
        has_bias: true,
    };
    let gelu_flops = ffn_gelu.flops(&[base_shape], &[output_shape]);

    // FFN with Relu
    let ffn_relu = Op::FusedFfnBlock {
        hidden_dim: 192,
        intermediate_dim: 768,
        activation: Activation::Relu,
        has_bias: true,
    };
    let relu_flops = ffn_relu.flops(&[base_shape], &[output_shape]);

    // FFN with Silu
    let ffn_silu = Op::FusedFfnBlock {
        hidden_dim: 192,
        intermediate_dim: 768,
        activation: Activation::Silu,
        has_bias: true,
    };
    let silu_flops = ffn_silu.flops(&[base_shape], &[output_shape]);

    // Relu has fewer FLOPs than Gelu and Silu
    assert!(relu_flops < gelu_flops);
    assert!(relu_flops < silu_flops);

    // Gelu and Silu should have the same FLOPs (both 8 per element)
    assert_eq!(gelu_flops, silu_flops);
}

// =============================================================================
// Shape-inference correctness regressions (Split / Concat / Linear / Reshape)
// =============================================================================

#[test]
fn split_infers_per_section_shapes() {
    let op = Op::Split { dim: 1, sections: vec![2, 3, 5] };
    let out = op.infer_shapes(&[&[4, 10, 8]]).unwrap();
    assert_eq!(out, vec![vec![4, 2, 8], vec![4, 3, 8], vec![4, 5, 8]]);
}

#[test]
fn split_rejects_section_sum_mismatch() {
    let op = Op::Split { dim: 1, sections: vec![2, 3] };
    assert!(op.infer_shapes(&[&[4, 10, 8]]).is_err());
}

#[test]
fn split_supports_negative_dim() {
    let op = Op::Split { dim: -1, sections: vec![3, 5] };
    let out = op.infer_shapes(&[&[4, 8]]).unwrap();
    assert_eq!(out, vec![vec![4, 3], vec![4, 5]]);
}

#[test]
fn concat_sums_the_axis_dimension() {
    let op = Op::Concat { axis: 0 };
    let out = op.infer_shapes(&[&[2, 4], &[3, 4], &[5, 4]]).unwrap();
    assert_eq!(out, vec![vec![10, 4]]);
}

#[test]
fn concat_rejects_non_axis_dim_mismatch() {
    let op = Op::Concat { axis: 0 };
    assert!(op.infer_shapes(&[&[2, 4], &[3, 5]]).is_err());
}

#[test]
fn linear_sets_last_dim_to_out_features() {
    let op = Op::Linear { out_features: 16, bias: true };
    let out = op.infer_shapes(&[&[2, 8, 32]]).unwrap();
    assert_eq!(out, vec![vec![2, 8, 16]]);
}

#[test]
fn reshape_infers_the_negative_one_dimension() {
    // total = 48, known dims product = 2*4 = 8, inferred = 48/8 = 6.
    let op = Op::Reshape { target: vec![2, -1, 4] };
    let out = op.infer_shapes(&[&[2, 6, 4]]).unwrap();
    assert_eq!(out, vec![vec![2, 6, 4]]);
}

#[test]
fn reshape_rejects_non_divisible_infer() {
    let op = Op::Reshape { target: vec![5, -1] };
    assert!(op.infer_shapes(&[&[2, 6, 4]]).is_err());
}

#[test]
fn reshape_rejects_multiple_infer_dims() {
    let op = Op::Reshape { target: vec![-1, -1] };
    assert!(op.infer_shapes(&[&[2, 6, 4]]).is_err());
}

#[test]
fn fused_transformer_flops_respect_activation() {
    use mlgraph::op::Activation;
    let inp = [&[1usize, 8, 32][..]];
    let out = [&[1usize, 8, 32][..]];
    let relu = Op::FusedTransformerLayer {
        num_heads: 4, head_dim: 8, hidden_dim: 32, intermediate_dim: 64,
        activation: Activation::Relu, has_bias: false,
    };
    let gelu = Op::FusedTransformerLayer {
        num_heads: 4, head_dim: 8, hidden_dim: 32, intermediate_dim: 64,
        activation: Activation::Gelu, has_bias: false,
    };
    // Previously the activation was hardcoded to Gelu, so these were equal.
    assert_ne!(
        relu.flops(&inp, &out),
        gelu.flops(&inp, &out),
        "activation must affect fused transformer FLOPS"
    );
}
