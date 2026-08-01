//! Element data types for tensors in the computation graph.
//!
//! These are compile-time metadata  -  `mlgraph` never touches actual tensor data.
//! The dtype determines byte-size calculations for bandwidth analysis.

use std::fmt;

/// Supported element data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DType {
    /// 16-bit floating point (IEEE 754 half-precision).
    F16,
    /// 16-bit brain floating point.
    Bf16,
    /// 32-bit floating point.
    F32,
    /// 64-bit floating point.
    F64,
    /// 8-bit unsigned integer.
    U8,
    /// 8-bit signed integer (quantized weights).
    I8,
    /// 32-bit signed integer.
    I32,
    /// 64-bit signed integer.
    I64,
    /// 4-bit integer (packed, 2 elements per byte).
    Int4,
}

impl DType {
    /// Size of one element in bytes.
    ///
    /// For sub-byte types like [`Int4`](DType::Int4), returns the effective
    /// bytes per element (0.5), rounded up to 1 for per-element calculations.
    /// Use [`byte_size_for_elements`](Self::byte_size_for_elements) for bulk
    /// calculations.
    pub fn byte_size(self) -> usize {
        match self {
            Self::F16 | Self::Bf16 => 2,
            Self::F32 | Self::I32 => 4,
            Self::F64 | Self::I64 => 8,
            Self::U8 | Self::I8 | Self::Int4 => 1,
            // Note: Int4 is rounded up here. Use byte_size_for_elements for bulk calculations.
        }
    }

    /// Total bytes needed to store `count` elements of this dtype.
    ///
    /// Handles sub-byte packing correctly (e.g., `Int4` packs 2 elements per byte).
    pub fn byte_size_for_elements(self, count: u64) -> u64 {
        match self {
            Self::Int4 => count.div_ceil(2),
            other => count.saturating_mul(other.byte_size() as u64),
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F16 => write!(f, "f16"),
            Self::Bf16 => write!(f, "bf16"),
            Self::F32 => write!(f, "f32"),
            Self::F64 => write!(f, "f64"),
            Self::U8 => write!(f, "u8"),
            Self::I8 => write!(f, "i8"),
            Self::I32 => write!(f, "i32"),
            Self::I64 => write!(f, "i64"),
            Self::Int4 => write!(f, "int4"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int4_packing() {
        assert_eq!(DType::Int4.byte_size_for_elements(1), 1);
        assert_eq!(DType::Int4.byte_size_for_elements(2), 1);
        assert_eq!(DType::Int4.byte_size_for_elements(3), 2);
        assert_eq!(DType::Int4.byte_size_for_elements(4), 2);
    }

    #[test]
    fn f16_byte_size() {
        assert_eq!(DType::F16.byte_size(), 2);
        assert_eq!(DType::F16.byte_size_for_elements(100), 200);
    }
}
