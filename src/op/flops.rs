use crate::op::{last_dim, Activation, Op};

pub(crate) fn flops_internal(op: &Op, input_shapes: &[&[usize]], output_shapes: &[&[usize]]) -> u64 {
    match op {
        Op::MatMul => flops_matmul(input_shapes),
        Op::Linear { out_features, bias } => flops_linear(input_shapes, *out_features, *bias),
        Op::LayerNorm { .. } | Op::Softmax { .. } => {
            if input_shapes.is_empty() || input_shapes[0].is_empty() { 0 } else { 5u64.saturating_mul(elements_saturating(input_shapes[0])) }
        }
        Op::Gelu | Op::Silu => {
            if input_shapes.is_empty() || input_shapes[0].is_empty() { 0 } else { 8u64.saturating_mul(elements_saturating(input_shapes[0])) }
        }
        Op::Relu | Op::ScalarMul { .. } => {
            if input_shapes.is_empty() || input_shapes[0].is_empty() { 0 } else { elements_saturating(input_shapes[0]) }
        }
        Op::Add | Op::Mul => {
            if output_shapes.is_empty() || output_shapes[0].is_empty() { 0 } else { elements_saturating(output_shapes[0]) }
        }
        Op::Reshape { .. } | Op::Transpose { .. } | Op::Split { .. } | Op::Concat { .. } => 0,
        Op::PatchEmbed { patch_size, in_channels, embed_dim, .. } => {
            flops_patch_embed(input_shapes, *patch_size, *in_channels, *embed_dim)
        }
        Op::FusedAttentionBlock { num_heads, head_dim, hidden_dim, has_bias } => {
            flops_fused_attention(input_shapes, *num_heads, *head_dim, *hidden_dim, *has_bias)
        }
        Op::FusedFfnBlock { hidden_dim, intermediate_dim, activation, has_bias } => {
            flops_fused_ffn(input_shapes, *hidden_dim, *intermediate_dim, *activation, *has_bias)
        }
        Op::FusedTransformerLayer { num_heads, head_dim, hidden_dim, intermediate_dim, activation, has_bias, .. } => {
            flops_fused_transformer(input_shapes, output_shapes, *num_heads, *head_dim, *hidden_dim, *intermediate_dim, *activation, *has_bias)
        }
    }
}

pub(crate) fn elements_saturating(shape: &[usize]) -> u64 {
    shape.iter().fold(1u64, |acc, &d| acc.saturating_mul(d as u64))
}

fn flops_matmul(input_shapes: &[&[usize]]) -> u64 {
    if input_shapes.len() < 2 {
        return 0;
    }
    let a = input_shapes[0];
    let b = input_shapes[1];
    if a.len() < 2 || b.len() < 2 { return 0; }
    
    let m = a[a.len() - 2] as u64;
    let k = last_dim(a);
    let n = last_dim(b);
    
    let mut batch = 1u64;
    for &d in &a[..a.len() - 2] {
        batch = batch.saturating_mul(d as u64);
    }
    batch = batch.max(1);
        
    let p = batch.saturating_mul(m).saturating_mul(k).saturating_mul(n);
    p.saturating_mul(2)
}

fn flops_linear(input_shapes: &[&[usize]], out_features: usize, bias: bool) -> u64 {
    if input_shapes.is_empty() {
        return 0;
    }
    let in_shape = input_shapes[0];
    if in_shape.is_empty() { return 0; }
    
    let batch_seq: u64 = in_shape[..in_shape.len() - 1]
        .iter()
        .fold(1u64, |acc, &d| acc.saturating_mul(d as u64));
        
    let in_features = last_dim(in_shape);
    
    let mut total = 2u64.saturating_mul(batch_seq).saturating_mul(in_features).saturating_mul(out_features as u64);
    if bias {
        total = total.saturating_add(batch_seq.saturating_mul(out_features as u64));
    }
    total
}

fn flops_patch_embed(input_shapes: &[&[usize]], patch_size: usize, in_channels: usize, embed_dim: usize) -> u64 {
    if input_shapes.is_empty() || input_shapes[0].len() < 4 || patch_size == 0 {
        return 0;
    }
    let batch = input_shapes[0][0] as u64;
    let h = input_shapes[0][2] as u64;
    let w = input_shapes[0][3] as u64;
    let num_patches = (h / patch_size as u64).saturating_mul(w / patch_size as u64);

    let kernel_flops = (in_channels.saturating_mul(patch_size).saturating_mul(patch_size)) as u64;
    
    let p = batch.saturating_mul(num_patches).saturating_mul(kernel_flops).saturating_mul(embed_dim as u64);
    p.saturating_mul(2)
}

fn flops_fused_attention(input_shapes: &[&[usize]], num_heads: usize, head_dim: usize, hidden_dim: usize, has_bias: bool) -> u64 {
    if input_shapes.is_empty() || input_shapes[0].len() < 2 {
        return 0;
    }
    let seq_len = input_shapes[0][input_shapes[0].len() - 2] as u64;
    let batch: u64 = input_shapes[0][..input_shapes[0].len() - 2]
        .iter()
        .map(|d| *d as u64)
        .product::<u64>()
        .max(1);
    let h = hidden_dim as u64;
    let nh = num_heads as u64;
    let hd = head_dim as u64;

    let mut total = 0u64;
    total = total.saturating_add(5u64.saturating_mul(batch).saturating_mul(seq_len).saturating_mul(h));
    total = total.saturating_add(2u64.saturating_mul(batch).saturating_mul(seq_len).saturating_mul(h).saturating_mul(3u64.saturating_mul(nh).saturating_mul(hd)));
    total = total.saturating_add(2u64.saturating_mul(batch).saturating_mul(nh).saturating_mul(seq_len).saturating_mul(seq_len).saturating_mul(hd));
    total = total.saturating_add(5u64.saturating_mul(batch).saturating_mul(nh).saturating_mul(seq_len).saturating_mul(seq_len));
    total = total.saturating_add(2u64.saturating_mul(batch).saturating_mul(nh).saturating_mul(seq_len).saturating_mul(seq_len).saturating_mul(hd));
    total = total.saturating_add(2u64.saturating_mul(batch).saturating_mul(seq_len).saturating_mul(nh.saturating_mul(hd)).saturating_mul(h));
    total = total.saturating_add(batch.saturating_mul(seq_len).saturating_mul(h));
    
    if has_bias {
        let bias_flops = batch.saturating_mul(seq_len).saturating_mul(3u64.saturating_mul(nh).saturating_mul(hd).saturating_add(h));
        total = total.saturating_add(bias_flops);
    }
    total
}

fn flops_fused_ffn(input_shapes: &[&[usize]], hidden_dim: usize, intermediate_dim: usize, activation: Activation, has_bias: bool) -> u64 {
    if input_shapes.is_empty() || input_shapes[0].len() < 2 {
        return 0;
    }
    let seq_len = input_shapes[0][input_shapes[0].len() - 2] as u64;
    let batch: u64 = input_shapes[0][..input_shapes[0].len() - 2]
        .iter()
        .map(|d| *d as u64)
        .product::<u64>()
        .max(1);
    let h = hidden_dim as u64;
    let inter = intermediate_dim as u64;

    let mut total = 0u64;
    total = total.saturating_add(5u64.saturating_mul(batch).saturating_mul(seq_len).saturating_mul(h));
    total = total.saturating_add(2u64.saturating_mul(batch).saturating_mul(seq_len).saturating_mul(h).saturating_mul(inter));
    
    let act_flops = match activation {
        Activation::Gelu | Activation::Silu => 8u64.saturating_mul(batch).saturating_mul(seq_len).saturating_mul(inter),
        Activation::Relu => batch.saturating_mul(seq_len).saturating_mul(inter),
    };
    total = total.saturating_add(act_flops);
    total = total.saturating_add(2u64.saturating_mul(batch).saturating_mul(seq_len).saturating_mul(inter).saturating_mul(h));
    total = total.saturating_add(batch.saturating_mul(seq_len).saturating_mul(h));
    
    if has_bias { 
        total = total.saturating_add(batch.saturating_mul(seq_len).saturating_mul(inter.saturating_add(h)));
    }
    total
}

#[allow(clippy::too_many_arguments)]
fn flops_fused_transformer(input_shapes: &[&[usize]], output_shapes: &[&[usize]], num_heads: usize, head_dim: usize, hidden_dim: usize, intermediate_dim: usize, activation: Activation, has_bias: bool) -> u64 {
    let attn = Op::FusedAttentionBlock {
        num_heads,
        head_dim,
        hidden_dim,
        has_bias,
    };
    let ffn = Op::FusedFfnBlock {
        hidden_dim,
        intermediate_dim,
        activation,
        has_bias,
    };
    flops_internal(&attn, input_shapes, output_shapes)
        .saturating_add(flops_internal(&ffn, input_shapes, output_shapes))
}
