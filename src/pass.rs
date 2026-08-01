//! Pass traits  -  the community extension points.
//!
//! All graph analysis and transformation is done through passes. Adding
//! a new analysis or fusion strategy requires implementing one trait.

use crate::error::Result;
use crate::graph::Graph;

/// A read-only analysis pass over the computation graph.
///
/// Implement this trait to add custom analysis (memory estimation,
/// bottleneck detection, roofline modeling, etc.).
///
/// # Example
///
/// ```rust,ignore
/// struct MyAnalysis;
///
/// impl AnalysisPass for MyAnalysis {
///     type Report = Vec<String>;
///
///     fn name(&self) -> &'static str { "my-analysis" }
///
///     fn analyze(&self, graph: &Graph) -> Result<Self::Report> {
///         Ok(graph.nodes().map(|n| n.name().to_string()).collect())
///     }
/// }
/// ```
pub trait AnalysisPass {
    /// The type of report this analysis produces.
    type Report;

    /// Human-readable name for logging.
    fn name(&self) -> &'static str;

    /// Run the analysis on the graph and produce a report.
    ///
    /// # Errors
    ///
    /// Returns an error if the analysis encounters invalid graph structure.
    fn analyze(&self, graph: &Graph) -> Result<Self::Report>;
}

/// A graph transformation pass that mutates the graph.
///
/// Implement this trait to add custom optimizations (fusion, pruning,
/// precision selection, etc.).
pub trait TransformPass {
    /// Human-readable name for logging.
    fn name(&self) -> &'static str;

    /// Apply this transformation to the graph, mutating it in place.
    ///
    /// Returns the number of transformations applied (e.g., number of
    /// fusions performed).
    ///
    /// # Errors
    ///
    /// Returns an error if the transformation encounters invalid graph
    /// structure or fails a precondition check.
    fn transform(&self, graph: &mut Graph) -> Result<usize>;
}
