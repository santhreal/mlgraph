//! Error types for `mlgraph`.
//!
//! Every error is actionable  -  the message tells you what went wrong and how
//! to fix it. No generic "something failed."

/// All errors that can occur in the graph optimizer.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// An operation received inputs with incompatible shapes.
    #[error(
        "shape mismatch in '{op_name}': {reason}. \
         Fix: verify the tensor shapes flowing into this operation."
    )]
    ShapeMismatch {
        /// The operation that detected the mismatch.
        op_name: String,
        /// What went wrong.
        reason: String,
    },

    /// A tensor or node referenced by ID does not exist in the graph.
    #[error(
        "reference to unknown {kind} id {id}. \
         Fix: ensure the id was returned by a previous graph operation."
    )]
    UnknownId {
        /// What kind of thing was missing ("node" or "tensor").
        kind: &'static str,
        /// The numeric id that was not found.
        id: u32,
    },

    /// A graph construction or optimization step was invalid.
    #[error("invalid graph operation: {reason}")]
    InvalidGraph {
        /// What went wrong.
        reason: String,
    },

    /// A fusion pattern could not be applied.
    #[error(
        "fusion '{pattern_name}' failed: {reason}. \
         Fix: ensure the subgraph matches the expected pattern."
    )]
    FusionFailed {
        /// The fusion pattern that failed.
        pattern_name: String,
        /// What went wrong.
        reason: String,
    },
}

/// Convenience result type.
pub type Result<T> = std::result::Result<T, Error>;
