use mlgraph::dtype::DType;
use mlgraph::graph::{Graph, NodeId, TensorId};
use mlgraph::op::Op;

#[test]
fn test_graph_tensor_id_limits() {
    let mut graph = Graph::new("limit_test");
    
    // Allocate a very large number of sequential operations to test graph topology limits
    // and internal ID capacity handling.
    let num_ops = 50_000;
    let mut current_tensor = graph.input("init", &[1, 10], DType::F32);
    
    for _ in 0..num_ops {
        let out = graph.node("relu", Op::Relu, &[current_tensor]);
        assert!(out.is_ok(), "Graph builder should not fail on normal operations");
        current_tensor = out.unwrap()[0];
    }
    
    // Graph should still be valid.
    assert_eq!(graph.num_nodes(), num_ops);
    assert_eq!(graph.num_tensors(), num_ops + 1);
    
    let order = graph.topological_order();
    assert_eq!(order.len(), num_ops);
}

#[test]
fn test_graph_fused_node_adversarial_mapping() {
    let mut graph = Graph::new("fusion_test");
    
    let t_in = graph.input("in", &[1, 10], DType::F32);
    let _t_out = graph.node("relu", Op::Relu, &[t_in]).unwrap()[0];
    
    assert_eq!(graph.num_nodes(), 1);
}
