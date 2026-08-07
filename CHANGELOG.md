# Changelog

## [0.1.3] - 2026-08-07

### Fixed
- BandwidthAnalysis missing-node / dtype / intensity edge cases; Split empty sections rejected.

### Changed
- Crate authors set to Santh noreply.


## [0.1.2] - 2026-08-02

### Fixed
- `BandwidthAnalysis` no longer silently truncates its report on cyclic graphs or silently drops dangling tensor references. Cyclic graphs now return `Error::InvalidGraph` naming the cycle, and unknown tensor ids return `Error::UnknownId`, instead of computing traffic from incomplete shape lists.
- Removed the dead, overflow-unsafe `elements` helper in `src/op/mod.rs`; `elements_saturating` is the single implementation.
- Fixed `Op::Split` shape inference to reject empty sections lists rather than returning empty output shapes.
- Fixed `BandwidthAnalysis` to return `Error::UnknownId` on missing node IDs during graph analysis instead of silently continuing.
- Updated `BandwidthAnalysis` arithmetic intensity calculation for nodes with zero HBM traffic and non-zero FLOPs to report infinite intensity (`f64::INFINITY`), correctly classifying fused in-SRAM operations as compute-bound.
- Updated `BandwidthAnalysis` to calculate HBM write traffic using the output tensor's specific data type rather than defaulting to the first input's data type.

### Added
- Adversarial test suite covering self-loops, two-node cycles, dangling inputs after node removal, phantom producers, and unknown-tensor fused mappings.

## [0.1.1] - 2026-07-31

### Fixed
- `Op::Linear` shape inference no longer uses `expect` on the output vector
  (deny-lint violation); the rank check is enforced with a panic-free
  let-else that returns `ShapeMismatch`.
- `Op::Transpose` duplicate-permutation detection is now O(n) via a seen-set
  instead of an O(n^2) per-index rescan; behavior is unchanged and covered by
  a new tail-duplicate regression test.

## [0.1.0] - 2026-04-12

### Added
- Initial release of mlgraph.
