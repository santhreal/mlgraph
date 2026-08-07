//! Gap test suite for mlgraph.
//!
//! Probes graph optimization boundaries and verifies behavior on empty or minimal graphs.

use mlgraph::graph::Graph;
use mlgraph::analysis::bandwidth::BandwidthAnalysis;
use mlgraph::pass::AnalysisPass;

/// Probes bandwidth analysis on an empty computation graph with zero nodes.
#[test]
fn test_gap_empty_graph_bandwidth_analysis() {
    let graph = Graph::new("empty");
    let report = BandwidthAnalysis.analyze(&graph).expect("empty graph analysis must succeed");
    assert_eq!(report.total_hbm_traffic, 0, "empty graph must have 0 HBM traffic");
    assert!(report.nodes.is_empty(), "empty graph must yield no node reports");
}
#[test]
fn test_split_rejects_empty_sections() {
    use mlgraph::op::Op;
    let op = Op::Split { dim: 0, sections: vec![] };
    assert!(op.infer_shapes(&[&[0]]).is_err(), "Split with empty sections must be rejected");
    assert!(op.infer_shapes(&[&[10]]).is_err(), "Split with empty sections must be rejected");
}

#[test]
fn test_bandwidth_mixed_precision_uses_output_dtype() {
    use mlgraph::dtype::DType;
    use mlgraph::graph::Graph;
    use mlgraph::op::Op;
    use mlgraph::analysis::bandwidth::BandwidthAnalysis;
    use mlgraph::pass::AnalysisPass;

    let mut graph = Graph::new("mixed_precision");
    let in_t = graph.input("in", &[2, 4], DType::F16);
    let out_ts = graph.node("linear", Op::Linear { out_features: 8, bias: false }, &[in_t]).unwrap();
    let out_t = out_ts[0];
    graph.tensor_mut(out_t).unwrap().set_dtype(DType::F32);

    let report = BandwidthAnalysis.analyze(&graph).expect("analysis must succeed");
    assert_eq!(report.nodes.len(), 1);
    let node = &report.nodes[0];
    // F16 input (2*4=8 elements * 2 bytes = 16 bytes read for input, plus weights)
    // F32 output (2*8=16 elements * 4 bytes = 64 bytes written for output)
    assert_eq!(node.hbm_write, 64, "Output traffic must be calculated using output tensor's F32 dtype (64 bytes), not F16 (32 bytes)");
}

#[test]
fn test_bandwidth_dangling_tensor_returns_error() {
    use mlgraph::dtype::DType;
    use mlgraph::graph::Graph;
    use mlgraph::op::Op;
    use mlgraph::analysis::bandwidth::BandwidthAnalysis;
    use mlgraph::pass::AnalysisPass;

    let mut graph = Graph::new("dangling");
    let in_t = graph.input("in", &[2, 4], DType::F16);
    let out_1 = graph.node("relu1", Op::Relu, &[in_t]).unwrap();
    let n1_id = graph.tensor(out_1[0]).unwrap().producer().unwrap();
    let _out_2 = graph.node("relu2", Op::Relu, &[out_1[0]]).unwrap();
    graph.remove_node(n1_id);

    let report = BandwidthAnalysis.analyze(&graph);
    assert!(report.is_err(), "BandwidthAnalysis must return error on dangling tensor reference");
}
