use mlgraph::dtype::DType;
use mlgraph::op::{Activation, Op};

#[test]
fn test_flops_integer_overflow() {
    let large_dim = usize::MAX;

    // MatMul overflow test
    let op = Op::MatMul;
    let shape_a = vec![1, large_dim, large_dim];
    let shape_b = vec![1, large_dim, large_dim];
    let out_shape = vec![1, large_dim, large_dim];

    let flops = op.flops(&[&shape_a, &shape_b], &[&out_shape]);
    assert_eq!(flops, u64::MAX, "Flops calculation should saturate to u64::MAX on integer overflow");

    // FusedAttentionBlock overflow test
    let op = Op::FusedAttentionBlock {
        num_heads: usize::MAX,
        head_dim: usize::MAX,
        hidden_dim: usize::MAX,
        has_bias: true,
    };
    let shape = vec![1, large_dim, large_dim];
    let flops = op.flops(&[&shape], &[&shape]);
    assert_eq!(flops, u64::MAX, "FusedAttentionBlock calculation should saturate to u64::MAX on integer overflow");
    
    // FusedFfnBlock overflow test
    let op = Op::FusedFfnBlock {
        hidden_dim: usize::MAX,
        intermediate_dim: usize::MAX,
        activation: Activation::Gelu,
        has_bias: true,
    };
    let flops = op.flops(&[&shape], &[&shape]);
    assert_eq!(flops, u64::MAX, "FusedFfnBlock calculation should saturate to u64::MAX on integer overflow");
}

#[test]
fn test_hbm_integer_overflow() {
    let large_dim = usize::MAX;
    
    let shape_in = vec![1, large_dim, large_dim];
    let shape_out = vec![1, large_dim, large_dim];

    use mlgraph::graph::Graph;
    use mlgraph::analysis::bandwidth::BandwidthAnalysis;
    use mlgraph::pass::AnalysisPass;
    
    let mut graph = Graph::new("overflow_test");
    let a = graph.input("a", &[shape_in[0], shape_in[1], shape_in[2]], DType::F32);
    let b = graph.input("b", &[shape_out[0], shape_out[1], shape_out[2]], DType::F32);
    
    let out = graph.node("matmul", Op::MatMul, &[a, b]);
    if let Ok(out) = out {
        graph.mark_output(out[0]);
        let analysis = BandwidthAnalysis;
        let res = analysis.analyze(&graph);
        assert!(res.is_ok(), "Bandwidth analysis should succeed without panicking on massive dimensions");
        let report = res.unwrap();
        // Saturated reading should be u64::MAX or near it
        assert!(report.total_hbm_traffic > 0, "Traffic should be recorded");
    }
}
