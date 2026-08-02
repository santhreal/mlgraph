//! Corrupted-topology adversarial tests for mlgraph.
//!
//! The graph IR is a DAG by contract, but the public mutation surface
//! (`add_fused_node`, `remove_node`, `update_tensor_producer`) can leave the
//! graph in a corrupted state: cycles, dangling tensor references, hijacked
//! producers. Analysis passes must refuse corrupted graphs with a specific
//! error instead of silently reporting partial or wrong numbers.

use mlgraph::analysis::bandwidth::BandwidthAnalysis;
use mlgraph::dtype::DType;
use mlgraph::error::Error;
use mlgraph::graph::Graph;
use mlgraph::op::Op;
use mlgraph::pass::AnalysisPass;

/// Build input -> relu -> relu and return the graph plus the tensor ids
/// `(t_in, t_mid, t_out)`.
fn two_relu_graph() -> (Graph, mlgraph::graph::TensorId, mlgraph::graph::TensorId, mlgraph::graph::TensorId) {
    let mut graph = Graph::new("corruption_fixture");
    let t_in = graph.input("in", &[1, 4], DType::F32);
    let t_mid = graph.node("relu1", Op::Relu, &[t_in]).unwrap()[0];
    let t_out = graph.node("relu2", Op::Relu, &[t_mid]).unwrap()[0];
    (graph, t_in, t_mid, t_out)
}

/// Why: `add_fused_node` accepts caller-chosen new tensor ids in its output
/// mapping. Mapping an output onto the fused node's own input tensor id
/// rewrites that tensor in place and creates a self-loop: the node both
/// produces and consumes the same tensor. Before the fix, `BandwidthAnalysis`
/// silently dropped every cyclic node from its report, under-counting HBM
/// traffic with no signal. It must now refuse with `InvalidGraph`.
#[test]
fn test_self_loop_via_fused_output_mapping_rejected() {
    let (mut graph, _t_in, t_mid, _t_out) = two_relu_graph();

    // Hostile mapping: the fused node consumes t_mid and "produces" t_mid.
    graph
        .add_fused_node("hostile", Op::Relu, &[t_mid], &[(t_mid, t_mid)])
        .unwrap();

    let res = BandwidthAnalysis.analyze(&graph);
    assert!(
        matches!(res, Err(Error::InvalidGraph { .. })),
        "self-loop graph must be refused with InvalidGraph, got {res:?}"
    );
}

/// Why: a two-node cycle is the canonical corrupted DAG. Topological order
/// visits only the acyclic prefix, so any analysis keyed off that order would
/// silently ignore cyclic nodes. The analysis must fail, not truncate.
#[test]
fn test_two_node_cycle_rejected_not_truncated() {
    let mut graph = Graph::new("cycle");
    let t_in = graph.input("in", &[1, 4], DType::F32);
    let t_a = graph.node("a", Op::Relu, &[t_in]).unwrap()[0];

    // Node "b" consumes t_a and overwrites t_a via the fused mapping, so
    // a -> t_a -> b -> t_a forms a cycle through the shared tensor.
    graph
        .add_fused_node("b", Op::Relu, &[t_a], &[(t_a, t_a)])
        .unwrap();

    let res = BandwidthAnalysis.analyze(&graph);
    match res {
        Err(Error::InvalidGraph { reason }) => {
            assert!(
                reason.contains("cycle"),
                "error must name the cycle, got: {reason}"
            );
        }
        other => panic!("cyclic graph must be rejected, got {other:?}"),
    }
}

/// Why: `remove_node` documents that it does not check whether removed
/// outputs are still consumed. A downstream node is left holding a dangling
/// input id. Shape lookup used to `filter_map` that hole away, computing
/// traffic from a truncated shape list. It must now surface `UnknownId`.
#[test]
fn test_dangling_input_after_remove_node_rejected() {
    let (mut graph, _t_in, _t_mid, _t_out) = two_relu_graph();

    // Find and remove relu1, orphaning relu2's input.
    let relu1_id = graph
        .nodes()
        .find(|n| n.name() == "relu1")
        .unwrap()
        .id();
    graph.remove_node(relu1_id);

    let res = BandwidthAnalysis.analyze(&graph);
    assert!(
        matches!(res, Err(Error::UnknownId { kind: "tensor", .. })),
        "dangling tensor reference must surface as UnknownId, got {res:?}"
    );
}

/// Why: removing a leaf node (no downstream consumers) is the legitimate use
/// of `remove_node`. The remaining graph is still a valid DAG and analysis
/// must succeed, proving the new validation rejects corruption without
/// rejecting legal mutation.
#[test]
fn test_remove_leaf_node_still_analyzes() {
    let (mut graph, _t_in, _t_mid, _t_out) = two_relu_graph();

    let relu2_id = graph
        .nodes()
        .find(|n| n.name() == "relu2")
        .unwrap()
        .id();
    graph.remove_node(relu2_id);

    let report = BandwidthAnalysis
        .analyze(&graph)
        .expect("acyclic remainder after leaf removal must analyze");
    assert_eq!(report.nodes.len(), 1, "only relu1 remains");
}

/// Why: `update_tensor_producer` can point a tensor at a producer that does
/// not exist. Topological order must still terminate (no hang, no panic) and
/// the node set it returns must be a subset of real nodes.
#[test]
fn test_phantom_producer_does_not_hang_topological_order() {
    let (mut graph, _t_in, t_mid, _t_out) = two_relu_graph();

    // Point t_mid at a fresh node id that was allocated but never inserted.
    let phantom = graph.alloc_node_id();
    graph.update_tensor_producer(t_mid, Some(phantom));

    let order = graph.topological_order();
    assert!(
        order.len() <= graph.num_nodes(),
        "topological order must not invent nodes"
    );
}

/// Why: a fused node whose output mapping references a tensor id that does
/// not exist must be rejected at construction (`UnknownId`), not stored as a
/// corrupt entry that later passes trip over.
#[test]
fn test_fused_mapping_to_unknown_tensor_rejected() {
    let (mut graph, _t_in, t_mid, _t_out) = two_relu_graph();

    // A freshly allocated id is validly typed but absent from the tensor map.
    let bogus = graph.alloc_tensor_id();
    let res = graph.add_fused_node("bad", Op::Relu, &[t_mid], &[(bogus, t_mid)]);
    assert!(
        matches!(res, Err(Error::UnknownId { kind: "tensor", .. })),
        "mapping from an unknown tensor must fail, got {res:?}"
    );
}

/// Why: degenerate "truncated" graphs (inputs declared, zero compute nodes)
/// are the boundary between valid and corrupt. Analysis must succeed with
/// zero traffic rather than erroring, because an empty DAG is a legal DAG.
#[test]
fn test_inputs_only_graph_analyzes_as_zero() {
    let mut graph = Graph::new("truncated");
    let _t = graph.input("in", &[2, 2], DType::F32);

    let report = BandwidthAnalysis
        .analyze(&graph)
        .expect("input-only graph is a valid empty DAG");
    assert_eq!(report.total_hbm_traffic, 0);
    assert!(report.nodes.is_empty());
}
