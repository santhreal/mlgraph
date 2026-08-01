use mlgraph::graph::Graph;
use mlgraph::dtype::DType;
use mlgraph::op::Op;
use std::sync::{Arc, Barrier, RwLock};
use std::thread;
use mlgraph::graph::TensorId;

#[test]
fn test_concurrent_graph_reads() {
    let mut graph = Graph::new("concurrent_read_test");
    
    // Build a simple graph
    let input1 = graph.input("in1", &[1, 10], DType::F32);
    let input2 = graph.input("in2", &[10, 20], DType::F32);
    
    let matmul_out = graph.node("matmul", Op::MatMul, &[input1, input2]).unwrap()[0];
    let relu_out = graph.node("relu", Op::Relu, &[matmul_out]).unwrap()[0];
    graph.mark_output(relu_out);

    let graph_arc = Arc::new(graph);
    let num_threads = 20;
    let barrier = Arc::new(Barrier::new(num_threads));
    
    let mut handles = vec![];
    
    for i in 0..num_threads {
        let g = Arc::clone(&graph_arc);
        let b = Arc::clone(&barrier);
        
        handles.push(thread::spawn(move || {
            b.wait(); // Force threads to start approximately at the same time
            
            // Perform read-only operations
            let order = g.topological_order();
            assert_eq!(order.len(), 2);
            
            let nodes: Vec<_> = g.nodes().collect();
            assert_eq!(nodes.len(), 2);
            
            let tensors: Vec<_> = g.tensors().collect();
            assert_eq!(tensors.len(), 4); // 2 inputs, 2 outputs
            
            let relu_node = g.node_by_id(order[1]).unwrap();
            assert_eq!(relu_node.name(), "relu");
            
            if i % 2 == 0 {
                assert!(g.tensor(tensors[0].id()).is_some());
            }
        }));
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_concurrent_graph_writes() {
    let graph = Arc::new(RwLock::new(Graph::new("concurrent_write_test")));
    let num_threads = 32;
    let barrier = Arc::new(Barrier::new(num_threads));
    
    let mut handles = vec![];
    
    // Have each thread rapidly add nodes and tensors.
    for i in 0..num_threads {
        let g = Arc::clone(&graph);
        let b = Arc::clone(&barrier);
        
        handles.push(thread::spawn(move || {
            b.wait();
            
            for j in 0..100 {
                let mut graph_lock = g.write().unwrap();
                let name = format!("t_{i}_{j}");
                let t = graph_lock.input(&name, &[1, 10], DType::F32);
                
                // Read operations mixed with writes
                let _ = graph_lock.node(&format!("relu_{i}_{j}"), Op::Relu, &[t]);
            }
        }));
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Verify graph is intact
    let graph_lock = graph.read().unwrap();
    assert_eq!(graph_lock.num_nodes(), 3200);
    assert_eq!(graph_lock.num_tensors(), 6400); // 3200 inputs + 3200 relu outputs
}

#[test]
fn test_concurrent_bandwidth_analysis() {
    use mlgraph::analysis::bandwidth::BandwidthAnalysis;
    use mlgraph::pass::AnalysisPass;
    
    let mut graph = Graph::new("concurrent_analysis_test");
    
    let input = graph.input("x", &[1, 20, 20], DType::F32);
    let ln_out = graph.node("ln", Op::LayerNorm { eps: 1e-5 }, &[input]).unwrap()[0];
    graph.mark_output(ln_out);

    let graph_arc = Arc::new(graph);
    let num_threads = 50;
    let barrier = Arc::new(Barrier::new(num_threads));
    
    let mut handles = vec![];
    
    for _ in 0..num_threads {
        let g = Arc::clone(&graph_arc);
        let b = Arc::clone(&barrier);
        
        handles.push(thread::spawn(move || {
            b.wait();
            
            let pass = BandwidthAnalysis;
            let report = pass.analyze(&g).unwrap();
            
            // F32 is 4 bytes. 
            // Input tensor elements = 400
            // Output tensor elements = 400
            // Total read = 400 * 4 + 2 * 20 * 4 (LayerNorm params) = 1600 + 160 = 1760 bytes
            // Total write = 400 * 4 = 1600 bytes
            // Total = 3360 bytes
            assert_eq!(report.total_hbm_traffic, 3360);
        }));
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
}
