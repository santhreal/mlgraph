# mlgraph  -  Technical Spec

## Overview

# mlgraph  -  bandwidth-aware computation graph optimizer  A Rust-native computation graph optimizer that analyzes and transforms ML model graphs to minimize HBM (High Bandwidth Memory) traffic through intelligent operation fusion.  ## The Problem  Modern GPU inference is **memory-bandwidth bound**, not compute-bound. Each unfused operation reads its inputs from HBM and writes its outputs back to HBM. A single transformer layer executes 10-15 separate GPU kernels, each causing an HBM round-trip. FlashAttention fuses the attention computation but leaves the surrounding operations unfused.  ## The Solution  `mlgraph` analyzes a model's computation graph, identifies fusible operation patterns (attention blocks, FFN blocks, full transformer layers), and produces optimized execution plans that minimize total HBM traffic.  ## Quick Start  ```rust use mlgraph::models::vit::{build_vit, ViTConfig}; use mlgraph::analysis::bandwidth::BandwidthAnalysis; use mlgraph::pass::AnalysisPass;  // Build a ViT-Tiny computation graph. let mut graph = build_vit(&ViTConfig::tiny()).unwrap();  // Analyze HBM bandwidth usage. let report = BandwidthAnalysis.analyze(&graph).unwrap(); println!("Total HBM traffic: {} bytes", report.total_hbm_traffic); ```  ## Extension Points  - **[`pass::AnalysisPass`]**  -  add custom analysis (roofline model, latency estimation) - **[`pass::TransformPass`]**  -  add custom optimizations (new fusion patterns, precision selection) - **[`models`]**  -  add pre-built model graphs for benchmarking  ## Architecture  ```text Model definition ──→ Graph IR ──→ Analysis passes ──→ Reports (ViT, BERT)      │ ├──→ Fusion passes ──→ Optimized Graph ──→ Analysis │ └──→ Emitter (future) ──→ CubeCL kernels ```

## Architecture

The crate is organized into the following public modules:

- `analysis`
- `dtype`
- `error`
- `fusion`
- `graph`
- `models`
- `op`
- `pass`

## Guarantees

- `#![forbid(unsafe_code)]` where applicable; see `src/lib.rs` for the exact lint preamble.
- All public types have doc comments.
- Error messages are actionable where applicable.

## Public API Summary

Key entry points are exported from `src/lib.rs` via `pub mod` and `pub use` re-exports.
Consult the module-level documentation in each source file for function signatures and usage examples.

## Error Handling

- `Error`
