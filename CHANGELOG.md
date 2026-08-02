# Changelog

## [0.1.2] - 2026-08-02

### Fixed
- `BandwidthAnalysis` no longer silently truncates its report on cyclic graphs or silently drops dangling tensor references. Cyclic graphs now return `Error::InvalidGraph` naming the cycle, and unknown tensor ids return `Error::UnknownId`, instead of computing traffic from incomplete shape lists.
- Removed the dead, overflow-unsafe `elements` helper in `src/op/mod.rs`; `elements_saturating` is the single implementation.

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
