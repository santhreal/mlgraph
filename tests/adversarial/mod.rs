//! adversarial tests for mlgraph.
//! See TESTING.md for the Santh testing standard.

/// Adversarial MatMul checks
pub mod matmul;
/// Adversarial Transpose checks
pub mod transpose;
/// Adversarial Overflow checks
pub mod overflow;
/// Adversarial Graph Limits checks
pub mod graph_limits;
/// Adversarial IO/OOM Mock checks
pub mod io_oom;
/// Adversarial Ops checks
pub mod ops;
/// Corrupted-topology checks (cycles, dangling refs, hostile fusion mappings)
pub mod corruption;
