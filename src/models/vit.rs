//! ViT (Vision Transformer) model graph builder.
//!
//! Constructs a computation graph for ViT variants (Tiny, Small, Base)
//! as defined in "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020).

use crate::dtype::DType;
use crate::error::Result;
use crate::graph::{Graph, TensorId};
use crate::op::Op;

/// Configuration for a Vision Transformer variant.
#[derive(Debug, Clone)]
pub struct ViTConfig {
    /// Name of this variant (e.g., "ViT-Tiny").
    pub name: String,
    /// Input image height and width (assumed square).
    pub image_size: usize,
    /// Patch size (e.g., 16).
    pub patch_size: usize,
    /// Number of input channels (e.g., 3 for RGB).
    pub in_channels: usize,
    /// Hidden dimension (embedding dim).
    pub hidden_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Intermediate FFN dimension (usually 4 × hidden_dim).
    pub mlp_dim: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of output classes.
    pub num_classes: usize,
    /// Element data type.
    pub dtype: DType,
    /// Batch size.
    pub batch_size: usize,
}

impl ViTConfig {
    /// ViT-Tiny: 5.7M parameters, 192 hidden, 3 heads, 12 layers.
    pub fn tiny() -> Self {
        Self {
            name: "ViT-Tiny".into(),
            image_size: 224,
            patch_size: 16,
            in_channels: 3,
            hidden_dim: 192,
            num_heads: 3,
            mlp_dim: 768,
            num_layers: 12,
            num_classes: 1000,
            dtype: DType::F16,
            batch_size: 1,
        }
    }

    /// ViT-Small: 22M parameters, 384 hidden, 6 heads, 12 layers.
    pub fn small() -> Self {
        Self {
            name: "ViT-Small".into(),
            image_size: 224,
            patch_size: 16,
            in_channels: 3,
            hidden_dim: 384,
            num_heads: 6,
            mlp_dim: 1536,
            num_layers: 12,
            num_classes: 1000,
            dtype: DType::F16,
            batch_size: 1,
        }
    }

    /// ViT-Base: 86M parameters, 768 hidden, 12 heads, 12 layers.
    pub fn base() -> Self {
        Self {
            name: "ViT-Base".into(),
            image_size: 224,
            patch_size: 16,
            in_channels: 3,
            hidden_dim: 768,
            num_heads: 12,
            mlp_dim: 3072,
            num_layers: 12,
            num_classes: 1000,
            dtype: DType::F16,
            batch_size: 1,
        }
    }

    /// Head dimension (hidden_dim / num_heads).
    pub fn head_dim(&self) -> usize {
        self.hidden_dim / self.num_heads
    }

    /// Number of patches.
    pub fn num_patches(&self) -> usize {
        (self.image_size / self.patch_size) * (self.image_size / self.patch_size)
    }

    /// Sequence length (patches + CLS token).
    pub fn seq_len(&self) -> usize {
        self.num_patches() + 1
    }
}

/// Build a complete ViT computation graph from a configuration.
///
/// The graph includes:
/// - Patch embedding (conv2d tokenization + CLS + position embedding)
/// - N transformer layers (attention block + FFN block each)
/// - Classification head (LayerNorm + Linear)
///
/// # Errors
///
/// Returns an error if shape inference fails during graph construction.
pub fn build_vit(config: &ViTConfig) -> Result<Graph> {
    if config.patch_size == 0 {
        return Err(crate::error::Error::InvalidGraph { reason: "patch_size must be > 0".to_string() });
    }
    if config.in_channels == 0 {
        return Err(crate::error::Error::InvalidGraph { reason: "in_channels must be > 0".to_string() });
    }
    if config.hidden_dim == 0 {
        return Err(crate::error::Error::InvalidGraph { reason: "hidden_dim must be > 0".to_string() });
    }
    if config.num_heads == 0 {
        return Err(crate::error::Error::InvalidGraph { reason: "num_heads must be > 0".to_string() });
    }
    if config.hidden_dim % config.num_heads != 0 {
        return Err(crate::error::Error::InvalidGraph {
            reason: format!("hidden_dim ({}) must be divisible by num_heads ({})", config.hidden_dim, config.num_heads),
        });
    }
    if config.image_size % config.patch_size != 0 {
        return Err(crate::error::Error::InvalidGraph {
            reason: format!("image_size ({}) must be divisible by patch_size ({})", config.image_size, config.patch_size),
        });
    }
    if config.batch_size == 0 {
        return Err(crate::error::Error::InvalidGraph { reason: "batch_size must be > 0".to_string() });
    }
    if config.mlp_dim == 0 {
        return Err(crate::error::Error::InvalidGraph { reason: "mlp_dim must be > 0".to_string() });
    }
    if config.num_layers == 0 {
        return Err(crate::error::Error::InvalidGraph { reason: "num_layers must be > 0".to_string() });
    }
    if config.num_classes == 0 {
        return Err(crate::error::Error::InvalidGraph { reason: "num_classes must be > 0".to_string() });
    }
    let mut g = Graph::new(&config.name);
    // Input image.
    let image = g.input(
        "image",
        &[config.batch_size, config.in_channels, config.image_size, config.image_size],
        config.dtype,
    );

    // Patch embedding → [B, seq_len, hidden_dim].
    let embedded = g.node(
        "patch_embed",
        Op::PatchEmbed {
            patch_size: config.patch_size,
            in_channels: config.in_channels,
            embed_dim: config.hidden_dim,
        },
        &[image],
    )?;
    let mut x = embedded[0];

    // Transformer layers.
    for layer_idx in 0..config.num_layers {
        let prefix = format!("L{layer_idx}");
        x = build_transformer_layer(&mut g, &prefix, x, config)?;
    }

    // Classification head.
    let ln_final = g.node(
        "head.ln",
        Op::LayerNorm { eps: 1e-6 },
        &[x],
    )?;
    let logits = g.node(
        "head.linear",
        Op::Linear {
            out_features: config.num_classes,
            bias: true,
        },
        &[ln_final[0]],
    )?;

    g.mark_output(logits[0]);
    Ok(g)
}

/// Build a single transformer layer (attention block + FFN block).
fn build_transformer_layer(
    g: &mut Graph,
    prefix: &str,
    x: TensorId,
    config: &ViTConfig,
) -> Result<TensorId> {
    let head_dim = config.head_dim();
    let b = config.batch_size;
    let s = config.seq_len();
    let h = config.hidden_dim;
    let nh = config.num_heads;

    // ── Attention block ─────────────────────────────────────────────

    // LayerNorm
    let ln1 = g.node(
        &format!("{prefix}.attn.ln"),
        Op::LayerNorm { eps: 1e-6 },
        &[x],
    )?[0];

    // QKV projection: [B, S, H] → [B, S, 3H]
    let qkv = g.node(
        &format!("{prefix}.attn.qkv"),
        Op::Linear {
            out_features: 3 * h,
            bias: true,
        },
        &[ln1],
    )?[0];

    // Split into Q, K, V: each [B, S, H]
    let qkv_split = g.node(
        &format!("{prefix}.attn.split"),
        Op::Split {
            dim: 2,
            sections: vec![h, h, h],
        },
        &[qkv],
    )?;
    let (q_flat, k_flat, v_flat) = (qkv_split[0], qkv_split[1], qkv_split[2]);

    // Reshape each to [B, S, num_heads, head_dim]
    let q_reshaped = g.node(
        &format!("{prefix}.attn.q.reshape"),
        Op::Reshape {
            target: vec![b as isize, s as isize, nh as isize, head_dim as isize],
        },
        &[q_flat],
    )?[0];
    let k_reshaped = g.node(
        &format!("{prefix}.attn.k.reshape"),
        Op::Reshape {
            target: vec![b as isize, s as isize, nh as isize, head_dim as isize],
        },
        &[k_flat],
    )?[0];
    let v_reshaped = g.node(
        &format!("{prefix}.attn.v.reshape"),
        Op::Reshape {
            target: vec![b as isize, s as isize, nh as isize, head_dim as isize],
        },
        &[v_flat],
    )?[0];

    // Transpose to [B, num_heads, S, head_dim]
    let q = g.node(
        &format!("{prefix}.attn.q.transpose"),
        Op::Transpose {
            perm: vec![0, 2, 1, 3],
        },
        &[q_reshaped],
    )?[0];
    let k = g.node(
        &format!("{prefix}.attn.k.transpose"),
        Op::Transpose {
            perm: vec![0, 2, 1, 3],
        },
        &[k_reshaped],
    )?[0];
    let v = g.node(
        &format!("{prefix}.attn.v.transpose"),
        Op::Transpose {
            perm: vec![0, 2, 1, 3],
        },
        &[v_reshaped],
    )?[0];

    // K^T: [B, num_heads, head_dim, S]
    let k_t = g.node(
        &format!("{prefix}.attn.k_t"),
        Op::Transpose {
            perm: vec![0, 1, 3, 2],
        },
        &[k],
    )?[0];

    // Attention scores: Q @ K^T → [B, num_heads, S, S]
    let scores = g.node(
        &format!("{prefix}.attn.scores"),
        Op::MatMul,
        &[q, k_t],
    )?[0];

    // Scale by 1/√head_dim
    let scaled = g.node(
        &format!("{prefix}.attn.scale"),
        Op::ScalarMul {
            factor: (1.0 / (head_dim as f64).sqrt()) as f32,
        },
        &[scores],
    )?[0];

    // Softmax
    let attn_weights = g.node(
        &format!("{prefix}.attn.softmax"),
        Op::Softmax { dim: -1 },
        &[scaled],
    )?[0];

    // Context: attn @ V → [B, num_heads, S, head_dim]
    let context = g.node(
        &format!("{prefix}.attn.context"),
        Op::MatMul,
        &[attn_weights, v],
    )?[0];

    // Transpose back: [B, S, num_heads, head_dim]
    let context_t = g.node(
        &format!("{prefix}.attn.context_transpose"),
        Op::Transpose {
            perm: vec![0, 2, 1, 3],
        },
        &[context],
    )?[0];

    // Reshape: [B, S, H]
    let context_flat = g.node(
        &format!("{prefix}.attn.context_reshape"),
        Op::Reshape {
            target: vec![b as isize, s as isize, h as isize],
        },
        &[context_t],
    )?[0];

    // Output projection: [B, S, H] → [B, S, H]
    let proj = g.node(
        &format!("{prefix}.attn.out_proj"),
        Op::Linear {
            out_features: h,
            bias: true,
        },
        &[context_flat],
    )?[0];

    // Residual add
    let attn_out = g.node(
        &format!("{prefix}.attn.residual"),
        Op::Add,
        &[x, proj],
    )?[0];

    // ── FFN block ───────────────────────────────────────────────────

    // LayerNorm
    let ln2 = g.node(
        &format!("{prefix}.ffn.ln"),
        Op::LayerNorm { eps: 1e-6 },
        &[attn_out],
    )?[0];

    // Up projection: [B, S, H] → [B, S, MLP_DIM]
    let ffn_up = g.node(
        &format!("{prefix}.ffn.up"),
        Op::Linear {
            out_features: config.mlp_dim,
            bias: true,
        },
        &[ln2],
    )?[0];

    // GELU activation
    let ffn_act = g.node(
        &format!("{prefix}.ffn.gelu"),
        Op::Gelu,
        &[ffn_up],
    )?[0];

    // Down projection: [B, S, MLP_DIM] → [B, S, H]
    let ffn_down = g.node(
        &format!("{prefix}.ffn.down"),
        Op::Linear {
            out_features: h,
            bias: true,
        },
        &[ffn_act],
    )?[0];

    // Residual add
    let layer_out = g.node(
        &format!("{prefix}.ffn.residual"),
        Op::Add,
        &[attn_out, ffn_down],
    )?[0];

    Ok(layer_out)
}
