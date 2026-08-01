# Changelog

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
