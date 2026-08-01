use mlgraph::dtype::DType;
use mlgraph::graph::Graph;
use mlgraph::op::Op;

#[test]
fn test_memory_scaling_limits() {
    let mut huge_graph = Graph::new("oom_test");
    
    // Create an input with dimensions that are large but won't cause immediate OOM on modern 64-bit systems
    // just to test that internal `TensorMeta` size calculations handle it safely without panicking due to overflow.
    let dims = vec![usize::MAX / 2; 10]; 
    
    let t1 = huge_graph.input("A", &dims, DType::F32);
    let t1_meta = huge_graph.tensor(t1).unwrap();
    
    assert_eq!(t1_meta.byte_size(), u64::MAX, "byte_size should saturate rather than overflow and panic");
    
    // Build a graph up to a size that ensures the internal nodes/tensors hash maps scale correctly
    // up to a reasonable limit without OOMing the test environment.
    let mut current_tensor = t1;
    for i in 0..10_000 {
        let out = huge_graph.node(&format!("B{}", i), Op::Relu, &[current_tensor]);
        assert!(out.is_ok(), "Graph builder should not fail on large graphs");
        current_tensor = out.unwrap()[0];
    }
    
    assert_eq!(huge_graph.num_nodes(), 10_000);
}

#[test]
fn test_oom_many_graph_edges() {
    let mut graph = Graph::new("dense_graph");
    
    // Create a fully connected dense dependency graph to test vector capacity limits.
    let mut inputs = Vec::new();
    for i in 0..1000 {
        inputs.push(graph.input(&format!("in_{i}"), &[1, 10], DType::F32));
    }
    
    // Make 500 nodes, each consuming ALL previous nodes via Concat to heavily stress topological sorting.
    for i in 0..500 {
        let node_inputs = inputs.clone();
        let out = graph.node(&format!("concat_{i}"), Op::Concat { axis: 0 }, &node_inputs);
        assert!(out.is_ok());
        inputs.push(out.unwrap()[0]);
    }
    
    // Verify topological order still succeeds with O(N^2) edges
    let order = graph.topological_order();
    assert_eq!(order.len(), 500);
}
