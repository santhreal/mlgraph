use crate::dtype::DType;
use crate::op::{last_dim, Op};

pub(crate) fn hbm_bytes_read_internal(op: &Op, input_shapes: &[&[usize]], dtype: DType) -> u64 {
    let input_elements: u64 = input_shapes.iter().fold(0u64, |acc, s| acc.saturating_add(crate::op::flops::elements_saturating(s)));

    match op {
        Op::Linear { out_features, bias } => hbm_bytes_read_linear(input_shapes, dtype, *out_features, *bias, input_elements),
        Op::LayerNorm { .. } => hbm_bytes_read_layer_norm(input_shapes, dtype, input_elements),
        Op::PatchEmbed { patch_size, in_channels, embed_dim, .. } => hbm_bytes_read_patch_embed(input_shapes, dtype, *patch_size, *in_channels, *embed_dim, input_elements),
        Op::FusedAttentionBlock { num_heads, head_dim, hidden_dim, has_bias } => hbm_bytes_read_fused_attention(dtype, *num_heads, *head_dim, *hidden_dim, *has_bias, input_elements),
        Op::FusedFfnBlock { hidden_dim, intermediate_dim, has_bias, .. } => hbm_bytes_read_fused_ffn(dtype, *hidden_dim, *intermediate_dim, *has_bias, input_elements),
        Op::FusedTransformerLayer { num_heads, head_dim, hidden_dim, intermediate_dim, has_bias, .. } => hbm_bytes_read_fused_transformer(input_elements, dtype, *num_heads, *head_dim, *hidden_dim, *intermediate_dim, *has_bias),
        Op::Concat { .. } => {
            // Sum the bytes of all input tensors
            input_shapes.iter().map(|s| dtype.byte_size_for_elements(crate::op::flops::elements_saturating(s))).sum()
        },
        _ => dtype.byte_size_for_elements(input_elements),
    }
}

fn hbm_bytes_read_linear(input_shapes: &[&[usize]], dtype: DType, out_features: usize, bias: bool, input_elements: u64) -> u64 {
    if input_shapes.is_empty() { return 0; }
    let in_features = last_dim(input_shapes[0]);
    let weight_elems = in_features.saturating_mul(out_features as u64);
    let bias_elems = if bias { out_features as u64 } else { 0 };
    dtype.byte_size_for_elements(input_elements.saturating_add(weight_elems).saturating_add(bias_elems))
}

fn hbm_bytes_read_layer_norm(input_shapes: &[&[usize]], dtype: DType, input_elements: u64) -> u64 {
    if input_shapes.is_empty() { return 0; }
    let norm_dim = last_dim(input_shapes[0]);
    dtype.byte_size_for_elements(input_elements.saturating_add(2u64.saturating_mul(norm_dim)))
}

fn hbm_bytes_read_patch_embed(input_shapes: &[&[usize]], dtype: DType, patch_size: usize, in_channels: usize, embed_dim: usize, input_elements: u64) -> u64 {
    if input_shapes.is_empty() || input_shapes[0].len() < 4 || patch_size == 0 {
        return 0;
    }
    let weight_elems = (in_channels.saturating_mul(patch_size).saturating_mul(patch_size).saturating_mul(embed_dim)) as u64;
    let h = input_shapes[0][2] as u64;
    let w = input_shapes[0][3] as u64;
    let num_patches = (h / patch_size as u64).saturating_mul(w / patch_size as u64);
    // Position and class embeddings are static parameters read once per forward
    // pass; like the patch weights they are not scaled by batch size.
    let pos_embed_elems = (num_patches.saturating_add(1)).saturating_mul(embed_dim as u64);
    let cls_elems = embed_dim as u64;

    let total = input_elements.saturating_add(weight_elems).saturating_add(pos_embed_elems).saturating_add(cls_elems);
    dtype.byte_size_for_elements(total)
}

fn hbm_bytes_read_fused_attention(dtype: DType, num_heads: usize, head_dim: usize, hidden_dim: usize, has_bias: bool, input_elements: u64) -> u64 {
    let activation_elems = input_elements;
    let qkv_weight = (hidden_dim as u64).saturating_mul(3u64.saturating_mul(num_heads as u64).saturating_mul(head_dim as u64));
    let out_weight = ((num_heads as u64).saturating_mul(head_dim as u64)).saturating_mul(hidden_dim as u64);
    let ln_params = 2u64.saturating_mul(hidden_dim as u64);
    let bias_elems = if has_bias {
        3u64.saturating_mul(num_heads as u64).saturating_mul(head_dim as u64).saturating_add(hidden_dim as u64)
    } else {
        0
    };
    let total = activation_elems.saturating_add(qkv_weight).saturating_add(out_weight).saturating_add(ln_params).saturating_add(bias_elems);
    dtype.byte_size_for_elements(total)
}

fn hbm_bytes_read_fused_ffn(dtype: DType, hidden_dim: usize, intermediate_dim: usize, has_bias: bool, input_elements: u64) -> u64 {
    let activation_elems = input_elements;
    let up_weight = (hidden_dim as u64).saturating_mul(intermediate_dim as u64);
    let down_weight = (intermediate_dim as u64).saturating_mul(hidden_dim as u64);
    let ln_params = 2u64.saturating_mul(hidden_dim as u64);
    let bias_elems = if has_bias {
        (intermediate_dim as u64).saturating_add(hidden_dim as u64)
    } else {
        0
    };
    let total = activation_elems.saturating_add(up_weight).saturating_add(down_weight).saturating_add(ln_params).saturating_add(bias_elems);
    dtype.byte_size_for_elements(total)
}

fn hbm_bytes_read_fused_transformer(input_elems: u64, dtype: DType, num_heads: usize, head_dim: usize, hidden_dim: usize, intermediate_dim: usize, has_bias: bool) -> u64 {
    let qkv_weight = (hidden_dim as u64).saturating_mul(3u64.saturating_mul(num_heads as u64).saturating_mul(head_dim as u64));
    let out_weight = ((num_heads as u64).saturating_mul(head_dim as u64)).saturating_mul(hidden_dim as u64);
    let up_weight = (hidden_dim as u64).saturating_mul(intermediate_dim as u64);
    let down_weight = (intermediate_dim as u64).saturating_mul(hidden_dim as u64);
    let ln_params = 4u64.saturating_mul(hidden_dim as u64);
    let bias_elems = if has_bias {
        3u64.saturating_mul(num_heads as u64).saturating_mul(head_dim as u64)
            .saturating_add(hidden_dim as u64)
            .saturating_add(intermediate_dim as u64)
            .saturating_add(hidden_dim as u64)
    } else {
        0
    };
    let total = input_elems.saturating_add(qkv_weight).saturating_add(out_weight).saturating_add(up_weight).saturating_add(down_weight).saturating_add(ln_params).saturating_add(bias_elems);
    dtype.byte_size_for_elements(total)
}

/// Estimate HBM bytes WRITTEN by this operation.
pub(crate) fn hbm_bytes_written(op: &Op, output_shapes: &[&[usize]], dtype: DType) -> u64 {
    if matches!(op, Op::Reshape { .. } | Op::Transpose { .. } | Op::Split { .. } | Op::Concat { .. }) {
        // Zero-copy operations write nothing explicitly
        return 0;
    }

    let output_elements: u64 = output_shapes.iter().fold(0u64, |acc, s| acc.saturating_add(crate::op::flops::elements_saturating(s)));
    dtype.byte_size_for_elements(output_elements)
}
