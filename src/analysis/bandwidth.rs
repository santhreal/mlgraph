//! Bandwidth analysis  -  estimate HBM traffic for every node in the graph.
//!
//! This is the core analysis that motivates kernel fusion. Each unfused
//! operation reads its inputs from HBM and writes its outputs to HBM.
//! Fused operations keep intermediates in SRAM, dramatically reducing
//! total HBM traffic.

use std::fmt;

use crate::error::Result;
use crate::graph::{Graph, NodeId};
use crate::pass::AnalysisPass;

/// Per-node bandwidth statistics.
#[derive(Debug, Clone)]
pub struct NodeBandwidth {
    /// The node this measurement belongs to.
    pub node_id: NodeId,
    /// Human-readable node name.
    pub node_name: String,
    /// Operation name.
    pub op_name: String,
    /// Bytes read from HBM.
    pub hbm_read: u64,
    /// Bytes written to HBM.
    pub hbm_write: u64,
    /// Total HBM traffic (read + write).
    pub hbm_total: u64,
    /// Floating-point operations.
    pub flops: u64,
    /// Arithmetic intensity: FLOPS / bytes moved.
    /// Higher = more compute-bound, lower = more bandwidth-bound.
    pub arithmetic_intensity: f64,
    /// Whether this is a fused operation.
    pub is_fused: bool,
}

/// Summary of bandwidth analysis across the entire graph.
#[derive(Debug, Clone)]
pub struct BandwidthReport {
    /// Per-node bandwidth breakdown.
    pub nodes: Vec<NodeBandwidth>,
    /// Total HBM bytes read across all nodes.
    pub total_hbm_read: u64,
    /// Total HBM bytes written across all nodes.
    pub total_hbm_write: u64,
    /// Total HBM traffic (read + write).
    pub total_hbm_traffic: u64,
    /// Total floating-point operations.
    pub total_flops: u64,
    /// Graph-wide arithmetic intensity.
    pub arithmetic_intensity: f64,
}

/// The bandwidth analysis pass.
///
/// Estimates HBM read/write traffic for every node based on tensor shapes,
/// dtypes, and operation semantics. Fused operations report dramatically
/// lower traffic because intermediates stay in SRAM.
pub struct BandwidthAnalysis;

impl AnalysisPass for BandwidthAnalysis {
    type Report = BandwidthReport;

    fn name(&self) -> &'static str {
        "bandwidth"
    }

    fn analyze(&self, graph: &Graph) -> Result<BandwidthReport> {
        let order = graph.topological_order();

        // A topological walk that visits fewer nodes than the graph contains
        // means the walk stalled on a cycle. Reporting bandwidth for only the
        // reachable prefix would silently under-count traffic, so refuse the
        // whole analysis instead.
        if order.len() != graph.num_nodes() {
            return Err(crate::error::Error::InvalidGraph {
                reason: format!(
                    "graph contains a cycle: topological order visited {} of {} nodes. \
                     Fix: remove the cyclic edge before running analysis.",
                    order.len(),
                    graph.num_nodes()
                ),
            });
        }

        let mut nodes = Vec::with_capacity(order.len());
        let mut total_read = 0_u64;
        let mut total_write = 0_u64;
        let mut total_flops = 0_u64;

        for nid in &order {
            let node = graph
                .node_by_id(*nid)
                .ok_or(crate::error::Error::UnknownId {
                    kind: "node",
                    id: nid.0,
                })?;

            // Dangling tensor references (e.g. left behind by `remove_node`
            // on a still-consumed producer) must fail loudly: shape lookup by
            // position matters, so dropping a missing input would compute
            // traffic from the wrong shape list with no signal.
            let input_shapes: Vec<Vec<usize>> = node
                .inputs()
                .iter()
                .map(|tid| {
                    graph
                        .tensor(*tid)
                        .map(|t| t.shape().to_vec())
                        .ok_or(crate::error::Error::UnknownId {
                            kind: "tensor",
                            id: tid.0,
                        })
                })
                .collect::<Result<Vec<_>>>()?;
            let output_shapes: Vec<Vec<usize>> = node
                .outputs()
                .iter()
                .map(|tid| {
                    graph
                        .tensor(*tid)
                        .map(|t| t.shape().to_vec())
                        .ok_or(crate::error::Error::UnknownId {
                            kind: "tensor",
                            id: tid.0,
                        })
                })
                .collect::<Result<Vec<_>>>()?;

            let input_refs: Vec<&[usize]> = input_shapes.iter().map(Vec::as_slice).collect();
            let output_refs: Vec<&[usize]> = output_shapes.iter().map(Vec::as_slice).collect();

            // Use the dtype of the first input tensor (or F16 as default).
            let in_dtype = node
                .inputs()
                .first()
                .and_then(|tid| graph.tensor(*tid))
                .map_or(crate::dtype::DType::F16, crate::graph::TensorMeta::dtype);
            let out_dtype = node
                .outputs()
                .first()
                .and_then(|tid| graph.tensor(*tid))
                .map_or(in_dtype, crate::graph::TensorMeta::dtype);

            let hbm_read = node.op().hbm_bytes_read(&input_refs, in_dtype);
            let hbm_write = node.op().hbm_bytes_written(&output_refs, out_dtype);
            let flops = node.op().flops(&input_refs, &output_refs);
            let hbm_total = hbm_read.saturating_add(hbm_write);
            let arithmetic_intensity = if hbm_total > 0 {
                flops as f64 / hbm_total as f64
            } else if flops > 0 {
                f64::INFINITY
            } else {
                0.0
            };

            nodes.push(NodeBandwidth {
                node_id: *nid,
                node_name: node.name().to_string(),
                op_name: node.op().name().to_string(),
                hbm_read,
                hbm_write,
                hbm_total,
                flops,
                arithmetic_intensity,
                is_fused: node.op().is_fused(),
            });

            total_read = total_read.saturating_add(hbm_read);
            total_write = total_write.saturating_add(hbm_write);
            total_flops = total_flops.saturating_add(flops);
        }

        let total_traffic = total_read.saturating_add(total_write);
        let arithmetic_intensity = if total_traffic > 0 {
            total_flops as f64 / total_traffic as f64
        } else if total_flops > 0 {
            f64::INFINITY
        } else {
            0.0
        };

        Ok(BandwidthReport {
            nodes,
            total_hbm_read: total_read,
            total_hbm_write: total_write,
            total_hbm_traffic: total_traffic,
            total_flops,
            arithmetic_intensity,
        })
    }
}

// ── Display formatting ──────────────────────────────────────────────────────

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

fn format_flops(flops: u64) -> String {
    if flops >= 1_000_000_000_000 {
        format!("{:.2} TFLOPS", flops as f64 / 1e12)
    } else if flops >= 1_000_000_000 {
        format!("{:.2} GFLOPS", flops as f64 / 1e9)
    } else if flops >= 1_000_000 {
        format!("{:.2} MFLOPS", flops as f64 / 1e6)
    } else if flops >= 1000 {
        format!("{:.2} KFLOPS", flops as f64 / 1e3)
    } else {
        format!("{flops} FLOPS")
    }
}

fn classify_intensity(intensity: f64) -> &'static str {
    if intensity.is_infinite() || intensity >= 10.0 {
        "compute-bound"
    } else if intensity < 1.0 {
        "bandwidth-bound"
    } else {
        "transitional"
    }
}

impl fmt::Display for BandwidthReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "┌──────────────────────────┬──────────┬──────────┬──────────────┬────────────┬──────────────────┐")?;
        writeln!(f, "│ Operation                │ HBM Read │ HBM Write│ HBM Total    │ FLOPs      │ Classification   │")?;
        writeln!(f, "├──────────────────────────┼──────────┼──────────┼──────────────┼────────────┼──────────────────┤")?;

        for node in &self.nodes {
            writeln!(
                f,
                "│ {:<24} │ {:>8} │ {:>8} │ {:>12} │ {:>10} │ {:<16} │",
                truncate(&node.node_name, 24),
                format_bytes(node.hbm_read),
                format_bytes(node.hbm_write),
                format_bytes(node.hbm_total),
                format_flops(node.flops),
                classify_intensity(node.arithmetic_intensity),
            )?;
        }

        writeln!(f, "├──────────────────────────┼──────────┼──────────┼──────────────┼────────────┼──────────────────┤")?;
        writeln!(
            f,
            "│ {:<24} │ {:>8} │ {:>8} │ {:>12} │ {:>10} │ {:<16} │",
            "TOTAL",
            format_bytes(self.total_hbm_read),
            format_bytes(self.total_hbm_write),
            format_bytes(self.total_hbm_traffic),
            format_flops(self.total_flops),
            classify_intensity(self.arithmetic_intensity),
        )?;
        writeln!(f, "└──────────────────────────┴──────────┴──────────┴──────────────┴────────────┴──────────────────┘")?;

        Ok(())
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    // Count and slice by characters, not bytes: byte slicing can land inside a
    // multibyte UTF-8 sequence and panic, and `max_len - 1` underflows at 0.
    if s.chars().count() <= max_len {
        s.to_string()
    } else if max_len == 0 {
        String::new()
    } else {
        // Reserve one column for the ellipsis.
        let truncated: String = s.chars().take(max_len - 1).collect();
        format!("{truncated}…")
    }
}
