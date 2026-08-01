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
