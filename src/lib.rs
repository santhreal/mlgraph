#![allow(
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::todo,
        clippy::unimplemented,
        clippy::panic
    )
)]
//! # mlgraph - bandwidth-aware computation graph optimizer
//!
//! ## Safe defaults
//!
//! **Input size:** No built-in cap. All computation graphs and node parameters are caller-constructed in memory. Memory consumption scales linearly with node and tensor count.
//!
//! **Recursion depth:** None. Graph topological sorting and pass analysis iterate over `Vec` collections without recursive calls.
//!
//! **Outbound network:** None. The crate performs pure in-memory graph transformations and contains no networking logic.
//!
//! **Process spawning:** None. The crate does not execute external processes or call `std::process::Command`.
//!
//! **Filesystem writes:** None. All analysis and fusion passes operate entirely on in-memory graph structures.
//!
//! **Credential exposure:** None. Computation graph structures handle tensor shape metadata and do not log or store secret values.
//!
//! A Rust-native computation graph optimizer that analyzes and transforms ML
//! model graphs to minimize HBM (High Bandwidth Memory) traffic through
//! intelligent operation fusion.
//!
//! ## The Problem
//!
//! Modern GPU inference is **memory-bandwidth bound**, not compute-bound.
//! Each unfused operation reads its inputs from HBM and writes its outputs
//! back to HBM. A single transformer layer executes 10-15 separate GPU
//! kernels, each causing an HBM round-trip. FlashAttention fuses the
//! attention computation but leaves the surrounding operations unfused.
//!
//! ## The Solution
//!
//! `mlgraph` analyzes a model's computation graph, identifies fusible
//! operation patterns (attention blocks, FFN blocks, full transformer layers),
//! and produces optimized execution plans that minimize total HBM traffic.
//!
//! ## Quick Start
//!
//! ```rust
//! use mlgraph::models::vit::{build_vit, ViTConfig};
//! use mlgraph::analysis::bandwidth::BandwidthAnalysis;
//! use mlgraph::pass::AnalysisPass;
//!
//! // Build a ViT-Tiny computation graph.
//! let mut graph = build_vit(&ViTConfig::tiny()).unwrap();
//!
//! // Analyze HBM bandwidth usage.
//! let report = BandwidthAnalysis.analyze(&graph).unwrap();
//! println!("Total HBM traffic: {} bytes", report.total_hbm_traffic);
//! ```
//!
//! ## Extension Points
//!
//! - **[`pass::AnalysisPass`]**  -  add custom analysis (roofline model, latency estimation)
//! - **[`pass::TransformPass`]**  -  add custom optimizations (new fusion patterns, precision selection)
//! - **[`models`]**  -  add pre-built model graphs for benchmarking
//!
//! ## Architecture
//!
//! ```text
//! Model definition ──→ Graph IR ──→ Analysis passes ──→ Reports
//!        (ViT, BERT)      │
//!                          ├──→ Fusion passes ──→ Optimized Graph ──→ Analysis
//!                          │
//!                          └──→ Emitter (future) ──→ CubeCL kernels
//! ```

#![warn(missing_docs, clippy::pedantic)]
#![allow(
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::module_name_repetitions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::needless_pass_by_value,
    clippy::many_single_char_names,
    clippy::doc_markdown,
    clippy::similar_names
)]

pub mod analysis;
pub mod dtype;
pub mod error;
pub mod fusion;
pub mod graph;
pub mod models;
/// Submodule for node operations
pub mod op;
pub mod pass;

pub use dtype::DType;
pub use error::{Error, Result};
pub use graph::{Graph, Node, NodeId, TensorId, TensorMeta};
pub use pass::{AnalysisPass, TransformPass};
