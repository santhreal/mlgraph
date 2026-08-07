use mlgraph::op::{Activation, Op};

#[test]
fn test_ops_empty_inputs() {
    let empty_shape: &[usize] = &[];
    let shape_1d: &[usize] = &[10];
    
    let ops_to_test = vec![
        Op::Linear { out_features: 10, bias: true },
        Op::LayerNorm { eps: 1e-5 },
        Op::Softmax { dim: -1 },
        Op::Relu,
        Op::Gelu,
        Op::Silu,
        Op::Add,
        Op::Mul,
        Op::ScalarMul { factor: 2.0 },
        Op::Reshape { target: vec![1, 10] },
        Op::Transpose { perm: vec![1, 0] },
        Op::Split { dim: 0, sections: vec![5, 5] },
        Op::Concat { axis: 0 },
        Op::PatchEmbed { patch_size: 16, in_channels: 3, embed_dim: 768 },
        Op::FusedAttentionBlock { num_heads: 12, head_dim: 64, hidden_dim: 768, has_bias: true },
        Op::FusedFfnBlock { hidden_dim: 768, intermediate_dim: 3072, activation: Activation::Gelu, has_bias: true },
        Op::FusedTransformerLayer { num_heads: 12, head_dim: 64, hidden_dim: 768, intermediate_dim: 3072, activation: Activation::Gelu, has_bias: true },
    ];
    
    for op in ops_to_test {
        // Test with no inputs at all
        let res = op.infer_shapes(&[]);
        assert!(res.is_err(), "infer_shapes should fail on zero inputs for {}", op.name());
        
        // Test with empty shape input
        let res2 = op.infer_shapes(&[empty_shape]);
        // Ops might pass or fail based on expected rank (e.g. Relu allows empty dims usually in PyTorch but here it returns Ok or Err)
        // Since we want strict adversarial testing, we assert it does not panic. 
        // We will match on it to make sure it's safely processed and the return value is explicitly handled.
        match res2 {
            Ok(sh) => {
                // If it passes, it shouldn't produce anything malformed
                assert!(!sh.is_empty(), "Resulting shapes array should not be completely empty if Ok");
            }
            Err(e) => {
                let err_str = e.to_string();
                assert!(err_str.contains("mismatch") || err_str.contains("inputs"), "Error should be a shape mismatch variant");
            }
        }
        
        // Test flops calculation with empty shapes
        let flops = op.flops(&[empty_shape], &[empty_shape]);
        assert_eq!(flops, 0, "{} should have 0 flops with empty inputs", op.name());
    }
}

#[test]
fn test_ops_all_zeros_dims() {
    let zero_shape: &[usize] = &[0, 0, 0];
    
    let op = Op::Linear { out_features: 0, bias: true };
    let flops = op.flops(&[zero_shape], &[zero_shape]);
    assert_eq!(flops, 0, "Linear should have 0 flops with zero-sized dims");
    
    let op = Op::FusedAttentionBlock { num_heads: 0, head_dim: 0, hidden_dim: 0, has_bias: true };
    let flops = op.flops(&[zero_shape], &[zero_shape]);
    assert_eq!(flops, 0, "FusedAttentionBlock should have 0 flops with zero-sized dims");
}

#[test]
fn test_ops_extreme_shapes_and_ranks() {
    // Rank 10 tensor
    let huge_rank = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let op = Op::MatMul;
    let flops = op.flops(&[&huge_rank, &huge_rank], &[&huge_rank]);
    assert!(flops > 0, "MatMul should compute flops on massive rank tensor");
    
    let op = Op::PatchEmbed { patch_size: 0, in_channels: 0, embed_dim: 0 };
    let flops = op.flops(&[&huge_rank], &[&huge_rank]);
    assert_eq!(flops, 0, "PatchEmbed should handle zero size patch cleanly");
    
    let infer_res = op.infer_shapes(&[&huge_rank]);
    assert!(infer_res.is_ok(), "PatchEmbed inference should succeed on large ranks");
    let out = infer_res.unwrap();
    assert_eq!(out[0], vec![1, 1, 0], "PatchEmbed inference returned incorrect dimensions for zero sizes");
}

#[test]
fn reshape_rejects_fully_specified_element_count_mismatch() {
    // Input [2,3] = 6 elements; a fully specified target of [4,5] = 20 elements
    // must be rejected, not silently accepted (which corrupted downstream sizes).
    let input: &[usize] = &[2, 3];
    let op = Op::Reshape { target: vec![4, 5] };
    let err = op.infer_shapes(&[input]).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("20 elements but input has 6"),
        "expected element-count mismatch, got: {msg}"
    );
}

#[test]
fn reshape_accepts_matching_count_and_infers_negative_one() {
    let input: &[usize] = &[2, 3];
    // Exact match.
    let exact = Op::Reshape { target: vec![6] }.infer_shapes(&[input]).unwrap();
    assert_eq!(exact[0], vec![6]);
    // Inferred dimension.
    let inferred = Op::Reshape { target: vec![-1, 2] }
        .infer_shapes(&[input])
        .unwrap();
    assert_eq!(inferred[0], vec![3, 2]);
}

#[test]
fn elementwise_add_rejects_mismatched_input_shapes() {
    let a: &[usize] = &[2, 3];
    let b: &[usize] = &[2, 4];
    let err = Op::Add.infer_shapes(&[a, b]).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("identical input shapes") && msg.contains("[2, 3]") && msg.contains("[2, 4]"),
        "expected identical-shape rejection, got: {msg}"
    );
}

#[test]
fn elementwise_mul_rejects_mismatched_input_shapes() {
    let a: &[usize] = &[8];
    let b: &[usize] = &[9];
    let err = Op::Mul.infer_shapes(&[a, b]).unwrap_err();
    assert!(
        err.to_string().contains("identical input shapes"),
        "expected identical-shape rejection, got: {err}"
    );
}

#[test]
fn elementwise_add_accepts_identical_input_shapes() {
    let a: &[usize] = &[2, 3];
    let b: &[usize] = &[2, 3];
    let out = Op::Add.infer_shapes(&[a, b]).unwrap();
    assert_eq!(out[0], vec![2, 3]);
}

#[test]
fn test_linear_rank1_input_infers_vector_output() {
    // Regression lock for the `last_mut().expect(...)` deny-lint violation:
    // Linear on a rank-1 input (a bare vector) must replace the single
    // dimension with out_features, panic-free.
    let op = Op::Linear { out_features: 7, bias: false };
    let shape = vec![13];
    let result = op.infer_shapes(&[&shape]).expect("rank-1 Linear must infer");
    assert_eq!(result[0], vec![7]);
}

#[test]
fn test_linear_rank0_input_is_rejected_not_panicking() {
    // A rank-0 input has no last dimension to rewrite; Linear must return a
    // ShapeMismatch error rather than panicking on `last_mut`.
    let op = Op::Linear { out_features: 7, bias: false };
    let shape: Vec<usize> = vec![];
    let result = op.infer_shapes(&[&shape]);
    assert!(
        matches!(result, Err(mlgraph::error::Error::ShapeMismatch { .. })),
        "rank-0 Linear must error, got {result:?}"
    );
}

#[test]
fn softmax_rejects_out_of_range_dimension() {
    let shape = vec![2, 3];
    let op_out_high = Op::Softmax { dim: 5 };
    assert!(op_out_high.infer_shapes(&[&shape]).is_err());

    let op_out_low = Op::Softmax { dim: -5 };
    assert!(op_out_low.infer_shapes(&[&shape]).is_err());

    let op_ok = Op::Softmax { dim: -1 };
    let res = op_ok.infer_shapes(&[&shape]).unwrap();
    assert_eq!(res[0], vec![2, 3]);
}

#[test]
fn layernorm_rejects_negative_or_nan_eps_and_rank0() {
    let shape = vec![2, 3];
    let op_neg = Op::LayerNorm { eps: -1.0 };
    assert!(op_neg.infer_shapes(&[&shape]).is_err());

    let op_nan = Op::LayerNorm { eps: f32::NAN };
    assert!(op_nan.infer_shapes(&[&shape]).is_err());

    let op_ok = Op::LayerNorm { eps: 1e-5 };
    let empty: Vec<usize> = vec![];
    assert!(op_ok.infer_shapes(&[&empty]).is_err());
}

#[test]
fn patchembed_rejects_channel_mismatch() {
    let op = Op::PatchEmbed { patch_size: 16, in_channels: 3, embed_dim: 768 };
    let bad_channels = vec![1, 4, 224, 224];
    assert!(op.infer_shapes(&[&bad_channels]).is_err());

    let good_channels = vec![1, 3, 224, 224];
    let res = op.infer_shapes(&[&good_channels]).unwrap();
    assert_eq!(res[0], vec![1, 197, 768]);
}

#[test]
fn linear_rejects_zero_out_features_and_multiple_inputs() {
    let shape = vec![2, 3];
    let op_zero = Op::Linear { out_features: 0, bias: true };
    assert!(op_zero.infer_shapes(&[&shape]).is_err());

    let op_valid = Op::Linear { out_features: 10, bias: true };
    assert!(op_valid.infer_shapes(&[&shape, &shape]).is_err());
}

#[test]
fn unary_ops_reject_multiple_inputs() {
    let shape = vec![2, 3];
    assert!(Op::Relu.infer_shapes(&[&shape, &shape]).is_err());
    assert!(Op::Gelu.infer_shapes(&[&shape, &shape]).is_err());
    assert!(Op::Silu.infer_shapes(&[&shape, &shape]).is_err());
    assert!(Op::ScalarMul { factor: 2.0 }.infer_shapes(&[&shape, &shape]).is_err());
}

#[test]
fn fused_ops_reject_mismatched_and_zero_dimensions() {
    let shape = vec![1, 197, 768];
    let bad_heads = Op::FusedAttentionBlock { num_heads: 0, head_dim: 64, hidden_dim: 768, has_bias: true };
    assert!(bad_heads.infer_shapes(&[&shape]).is_err());

    let head_mismatch = Op::FusedAttentionBlock { num_heads: 12, head_dim: 64, hidden_dim: 512, has_bias: true };
    assert!(head_mismatch.infer_shapes(&[&shape]).is_err());

    let input_mismatch = Op::FusedAttentionBlock { num_heads: 12, head_dim: 64, hidden_dim: 768, has_bias: true };
    let bad_shape = vec![1, 197, 512];
    assert!(input_mismatch.infer_shapes(&[&bad_shape]).is_err());

    let ffn_mismatch = Op::FusedFfnBlock { hidden_dim: 768, intermediate_dim: 3072, activation: Activation::Gelu, has_bias: true };
    assert!(ffn_mismatch.infer_shapes(&[&bad_shape]).is_err());
}

#[test]
fn build_vit_rejects_invalid_configurations() {
    use mlgraph::models::vit::{build_vit, ViTConfig};

    let mut cfg = ViTConfig::tiny();
    cfg.patch_size = 0;
    assert!(build_vit(&cfg).is_err());

    let mut cfg2 = ViTConfig::tiny();
    cfg2.num_heads = 0;
    assert!(build_vit(&cfg2).is_err());

    let mut cfg3 = ViTConfig::tiny();
    cfg3.hidden_dim = 100;
    cfg3.num_heads = 3;
    assert!(build_vit(&cfg3).is_err());

    let mut cfg4 = ViTConfig::tiny();
    cfg4.image_size = 225;
    cfg4.patch_size = 16;
    assert!(build_vit(&cfg4).is_err());
}
