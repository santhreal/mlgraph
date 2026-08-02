use crate::error::{Error, Result};

/// Submodule for calculating HBM bandwidth
pub mod hbm;
/// Submodule for calculating FLOPS
pub mod flops;

/// Supported activation functions for fused operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Activation {
    /// Rectified Linear Unit.
    Relu,
    /// Gaussian Error Linear Unit.
    Gelu,
    /// Sigmoid Linear Unit.
    Silu,
}

// f32 instead of f64, or use a custom Hash/Eq if needed, but we can just drop Eq/Hash since Op doesn't need to be hashed natively if we don't use it in sets.
// Actually Eq and Hash are needed by Node maybe? Let's drop Eq and Hash and see if it compiles, or use String representations, or ordered_float.
// Wait, Node is in a HashMap by ID, not by Op.
/// Computation operations available in the graph.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Op {
    /// Matrix multiplication.
    MatMul,
    /// Linear transformation (dense layer).
    Linear { 
        /// Output features dimension.
        out_features: usize, 
        /// Whether the layer has a bias vector.
        bias: bool 
    },
    /// Layer normalization.
    LayerNorm { 
        /// Epsilon value for numerical stability.
        eps: f32 
    },
    /// Softmax activation over a specific dimension.
    Softmax { 
        /// Dimension to apply softmax over.
        dim: isize 
    },
    /// Rectified Linear Unit activation.
    Relu,
    /// Gaussian Error Linear Unit activation.
    Gelu,
    /// Sigmoid Linear Unit activation.
    Silu,
    /// Element-wise addition.
    Add,
    /// Element-wise multiplication.
    Mul,
    /// Multiplication by a scalar value.
    ScalarMul { 
        /// Scalar factor.
        factor: f32 
    },
    /// Reshape tensor dimensions.
    Reshape { 
        /// Target shape (-1 means infer).
        target: Vec<isize> 
    },
    /// Transpose tensor dimensions.
    Transpose { 
        /// Permutation mapping.
        perm: Vec<usize> 
    },
    /// Split a tensor into multiple sections.
    Split { 
        /// Dimension to split on.
        dim: isize, 
        /// Sizes of the individual sections.
        sections: Vec<usize> 
    },
    /// Concatenate multiple tensors.
    Concat { 
        /// Dimension to concatenate on.
        axis: isize 
    },
    /// Vision transformer patch embedding.
    PatchEmbed { 
        /// Size of spatial patches.
        patch_size: usize, 
        /// Number of input image channels.
        in_channels: usize, 
        /// Output embedding dimension.
        embed_dim: usize 
    },
    /// A fused attention block (MHA/GQA/MQA).
    FusedAttentionBlock { 
        /// Number of attention heads.
        num_heads: usize, 
        /// Dimension of each head.
        head_dim: usize, 
        /// Hidden dimension.
        hidden_dim: usize, 
        /// Whether projections have biases.
        has_bias: bool 
    },
    /// A fused feed-forward network block.
    FusedFfnBlock { 
        /// Hidden dimension (input/output).
        hidden_dim: usize, 
        /// Intermediate expanded dimension.
        intermediate_dim: usize, 
        /// Activation function used.
        activation: Activation, 
        /// Whether projections have biases.
        has_bias: bool 
    },
    /// A fully fused transformer layer (Attention + FFN).
    FusedTransformerLayer { 
        /// Number of attention heads.
        num_heads: usize, 
        /// Dimension of each head.
        head_dim: usize, 
        /// Hidden dimension (input/output).
        hidden_dim: usize, 
        /// Intermediate expanded dimension for FFN.
        intermediate_dim: usize, 
        /// Activation function for FFN.
        activation: Activation, 
        /// Whether projections have biases.
        has_bias: bool 
    },
}

impl Op {
    /// Return the string name of the op.
    pub fn name(&self) -> &str {
        match self {
            Op::MatMul => "matmul",
            Op::Linear { .. } => "linear",
            Op::LayerNorm { .. } => "layer_norm",
            Op::Softmax { .. } => "softmax",
            Op::Relu => "relu",
            Op::Gelu => "gelu",
            Op::Silu => "silu",
            Op::Add => "add",
            Op::Mul => "mul",
            Op::ScalarMul { .. } => "scalar_mul",
            Op::Reshape { .. } => "reshape",
            Op::Transpose { .. } => "transpose",
            Op::Split { .. } => "split",
            Op::Concat { .. } => "concat",
            Op::PatchEmbed { .. } => "patch_embed",
            Op::FusedAttentionBlock { .. } => "fused_attention_block",
            Op::FusedFfnBlock { .. } => "fused_ffn_block",
            Op::FusedTransformerLayer { .. } => "fused_transformer_layer",
        }
    }

    /// Returns true if this operation is a fused operation block.
    pub fn is_fused(&self) -> bool {
        matches!(self, Op::FusedAttentionBlock { .. } | Op::FusedFfnBlock { .. } | Op::FusedTransformerLayer { .. })
    }

    /// Return the number of outputs generated.
    pub fn num_outputs(&self) -> usize {
        match self {
            Op::Split { sections, .. } => sections.len(),
            _ => 1,
        }
    }
    
    /// Estimate output shapes given inputs.
    ///
    /// # Errors
    /// Returns [`Error::ShapeMismatch`] if input shapes are incompatible.
    pub fn infer_shapes(&self, input_shapes: &[&[usize]]) -> Result<Vec<Vec<usize>>> {
        if input_shapes.is_empty() {
            return Err(Error::ShapeMismatch { op_name: self.name().to_string(), reason: "No inputs provided".to_string() });
        }
        
        match self {
            Op::MatMul => {
                if input_shapes.len() != 2 {
                    return Err(Error::ShapeMismatch { op_name: self.name().to_string(), reason: "expected 2 inputs".to_string() });
                }
                let a = input_shapes[0];
                let b = input_shapes[1];
                if a.len() < 2 || b.len() < 2 {
                    return Err(Error::ShapeMismatch { op_name: self.name().to_string(), reason: "inputs must be at least 2-dimensional".to_string() });
                }
                
                let a_last = a[a.len() - 1];
                let b_prev_last = b[b.len() - 2];
                if a_last != b_prev_last {
                    return Err(Error::ShapeMismatch { 
                        op_name: self.name().to_string(), 
                        reason: format!("contraction dimension mismatch: a[-1]={a_last} != b[-2]={b_prev_last}")
                    });
                }
                
                let mut out_shape = a[..a.len() - 1].to_vec();
                out_shape.push(b[b.len() - 1]);
                Ok(vec![out_shape])
            },
            Op::Transpose { perm } => {
                let a = input_shapes[0];
                if perm.len() != a.len() {
                    return Err(Error::ShapeMismatch { op_name: self.name().to_string(), reason: "permutation length mismatch".to_string() });
                }
                let mut out_shape = vec![0; a.len()];
                let mut seen = vec![false; a.len()];
                for (i, p) in perm.iter().enumerate() {
                    if *p >= a.len() {
                        return Err(Error::ShapeMismatch { op_name: self.name().to_string(), reason: "permutation index out of bounds".to_string() });
                    }
                    // O(n) duplicate detection via a seen-set; the previous
                    // per-index rescan was O(n^2) in the rank.
                    if seen[*p] {
                        return Err(Error::ShapeMismatch { op_name: self.name().to_string(), reason: "permutation index appears multiple times, must be unique".to_string() });
                    }
                    seen[*p] = true;
                    out_shape[i] = a[*p];
                }
                Ok(vec![out_shape])
            },
            Op::PatchEmbed { patch_size, embed_dim, .. } => {
                let a = input_shapes[0];
                if a.len() < 4 {
                    return Err(Error::ShapeMismatch { op_name: self.name().to_string(), reason: "PatchEmbed requires rank 4 input".to_string() });
                }
                let b = a[0];
                let h = a[2];
                let w = a[3];
                let s = if *patch_size == 0 {
                    1
                } else {
                    (h / *patch_size) * (w / *patch_size) + 1
                };
                Ok(vec![vec![b, s, *embed_dim]])
            }
            Op::Linear { out_features, .. } => {
                // A dense layer maps the last dimension (in_features) to out_features
                // and leaves the leading (batch/sequence) dimensions unchanged.
                let a = input_shapes[0];
                if a.is_empty() {
                    return Err(Error::ShapeMismatch { op_name: self.name().to_string(), reason: "Linear requires a rank >= 1 input".to_string() });
                }
                let mut out = a.to_vec();
                // Rank is checked non-empty above; let-else keeps this
                // panic-free (the crate denies `expect`).
                let Some(last) = out.last_mut() else {
                    return Err(Error::ShapeMismatch { op_name: self.name().to_string(), reason: "Linear requires a rank >= 1 input".to_string() });
                };
                *last = *out_features;
                Ok(vec![out])
            }
            Op::Split { dim, sections } => {
                let a = input_shapes[0];
                let axis = normalize_axis(*dim, a.len()).ok_or_else(|| Error::ShapeMismatch {
                    op_name: self.name().to_string(),
                    reason: format!("split dim {dim} out of range for rank {}", a.len()),
                })?;
                let sections_sum: usize = sections.iter().copied().fold(0usize, usize::saturating_add);
                if sections_sum != a[axis] {
                    return Err(Error::ShapeMismatch {
                        op_name: self.name().to_string(),
                        reason: format!("split sections sum {sections_sum} != dim size {}", a[axis]),
                    });
                }
                let mut out = Vec::with_capacity(sections.len());
                for &section in sections {
                    let mut shape = a.to_vec();
                    shape[axis] = section;
                    out.push(shape);
                }
                Ok(out)
            }
            Op::Concat { axis } => {
                let a = input_shapes[0];
                let ax = normalize_axis(*axis, a.len()).ok_or_else(|| Error::ShapeMismatch {
                    op_name: self.name().to_string(),
                    reason: format!("concat axis {axis} out of range for rank {}", a.len()),
                })?;
                let mut total = 0usize;
                for s in input_shapes {
                    if s.len() != a.len() {
                        return Err(Error::ShapeMismatch { op_name: self.name().to_string(), reason: "concat inputs have differing ranks".to_string() });
                    }
                    for (i, (&d_ref, &d)) in a.iter().zip(s.iter()).enumerate() {
                        if i != ax && d_ref != d {
                            return Err(Error::ShapeMismatch { op_name: self.name().to_string(), reason: format!("concat inputs disagree on non-axis dim {i}: {d_ref} != {d}") });
                        }
                    }
                    total = total.saturating_add(s[ax]);
                }
                let mut out = a.to_vec();
                out[ax] = total;
                Ok(vec![out])
            }
            Op::Reshape { target } => {
                let a = input_shapes[0];
                let total: usize = a.iter().copied().fold(1usize, usize::saturating_mul);
                let mut known: usize = 1;
                let mut infer_positions = 0;
                for &dim in target {
                    if dim == -1 {
                        infer_positions += 1;
                    } else if dim < 0 {
                        return Err(Error::ShapeMismatch { op_name: self.name().to_string(), reason: format!("reshape target dim {dim} is invalid (only -1 may be negative)") });
                    } else {
                        known = known.saturating_mul(dim.unsigned_abs());
                    }
                }
                if infer_positions > 1 {
                    return Err(Error::ShapeMismatch { op_name: self.name().to_string(), reason: "reshape target has more than one -1".to_string() });
                }
                let inferred = if infer_positions == 1 {
                    if known == 0 || total % known != 0 {
                        return Err(Error::ShapeMismatch { op_name: self.name().to_string(), reason: format!("cannot infer -1 dim: {total} elements not divisible by product of known dims {known}") });
                    }
                    total / known
                } else {
                    // No inferred (-1) dim: the fully-specified target must preserve
                    // the element count. Without this check a reshape to a differently
                    // sized shape was silently accepted, corrupting every downstream
                    // memory-size calculation (Law 10).
                    if known != total {
                        return Err(Error::ShapeMismatch { op_name: self.name().to_string(), reason: format!("reshape target has {known} elements but input has {total}") });
                    }
                    0
                };
                let mut out_shape = Vec::with_capacity(target.len());
                for &dim in target {
                    if dim == -1 {
                        out_shape.push(inferred);
                    } else {
                        out_shape.push(dim.unsigned_abs());
                    }
                }
                Ok(vec![out_shape])
            }
            Op::Add | Op::Mul => {
                // Element-wise binary ops require identical input shapes: this
                // framework has no broadcasting. The old default arm silently
                // returned the first input's shape, hiding a real mismatch and
                // corrupting downstream element counts (Law 10).
                if input_shapes.len() != 2 {
                    return Err(Error::ShapeMismatch { op_name: self.name().to_string(), reason: format!("element-wise {} expects 2 inputs, got {}", self.name(), input_shapes.len()) });
                }
                let a = input_shapes[0];
                let b = input_shapes[1];
                if a != b {
                    return Err(Error::ShapeMismatch { op_name: self.name().to_string(), reason: format!("element-wise {} requires identical input shapes: {a:?} != {b:?}", self.name()) });
                }
                Ok(vec![a.to_vec()])
            }
            _ => {
                // Default to first input shape
                Ok(vec![input_shapes[0].to_vec()])
            }
        }
    }
    
    /// Calculate estimated FLOPS.
    pub fn flops(&self, input_shapes: &[&[usize]], output_shapes: &[&[usize]]) -> u64 {
        flops::flops_internal(self, input_shapes, output_shapes)
    }

    /// Calculate estimated read bytes from HBM.
    pub fn hbm_bytes_read(&self, input_shapes: &[&[usize]], dtype: crate::dtype::DType) -> u64 {
        hbm::hbm_bytes_read_internal(self, input_shapes, dtype)
    }

    /// Calculate estimated written bytes to HBM.
    pub fn hbm_bytes_written(&self, output_shapes: &[&[usize]], dtype: crate::dtype::DType) -> u64 {
        hbm::hbm_bytes_written(self, output_shapes, dtype)
    }
}

/// Normalize a possibly-negative axis (numpy/PyTorch convention: -1 == last)
/// against a tensor rank, returning `None` when it falls outside `[-rank, rank)`.
pub(crate) fn normalize_axis(axis: isize, rank: usize) -> Option<usize> {
    let r = rank as isize;
    let resolved = if axis < 0 { axis + r } else { axis };
    if resolved < 0 || resolved >= r {
        None
    } else {
        Some(resolved as usize)
    }
}

pub(crate) fn last_dim(shape: &[usize]) -> u64 {
    *shape.last().unwrap_or(&1) as u64
}
