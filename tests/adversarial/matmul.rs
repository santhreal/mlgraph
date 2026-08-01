use mlgraph::error::Error;
use mlgraph::op::Op;

#[test]
fn test_matmul_empty_inputs() {
    let op = Op::MatMul;
    let result = op.infer_shapes(&[]);
    assert!(matches!(result, Err(Error::ShapeMismatch { .. })));
}

#[test]
fn test_matmul_1d_inputs() {
    let op = Op::MatMul;
    let shape_a = vec![10];
    let shape_b = vec![10];
    let result = op.infer_shapes(&[&shape_a, &shape_b]);
    assert!(matches!(result, Err(Error::ShapeMismatch { .. })));
}

#[test]
fn test_matmul_incompatible_inner_dims() {
    let op = Op::MatMul;
    let shape_a = vec![1, 10, 20];
    let shape_b = vec![1, 30, 40];
    let result = op.infer_shapes(&[&shape_a, &shape_b]);
    
    assert!(matches!(result, Err(Error::ShapeMismatch { .. })));
}

#[test]
fn test_matmul_zero_dims() {
    let op = Op::MatMul;
    let shape_a = vec![0, 10];
    let shape_b = vec![10, 0];
    let result = op.infer_shapes(&[&shape_a, &shape_b]).unwrap();
    assert_eq!(result[0], vec![0, 0]);
}

#[test]
fn test_matmul_extreme_dims() {
    let op = Op::MatMul;
    let shape_a = vec![usize::MAX, 10];
    let shape_b = vec![10, usize::MAX];
    let result = op.infer_shapes(&[&shape_a, &shape_b]).unwrap();
    assert_eq!(result[0], vec![usize::MAX, usize::MAX]);
}

#[test]
fn test_matmul_flops_extreme_dims() {
    let op = Op::MatMul;
    let shape_a = vec![1, 10];
    let shape_b = vec![10, 1];
    let flops = op.flops(&[&shape_a, &shape_b], &[&[1, 1]]);
    assert_eq!(flops, 20); // 2 * M * N * K = 2 * 1 * 1 * 10 = 20

    // Flops for extreme dims (wrapping will occur naturally or panic based on profile, 
    // but in release with aggressive u64 it might wrap. The key is it doesn't crash inappropriately).
    let shape_a_large = vec![1, 1 << 30];
    let shape_b_large = vec![1 << 30, 1];
    let large_flops = op.flops(&[&shape_a_large, &shape_b_large], &[&[1, 1]]);
    assert_eq!(large_flops, 2 * 1 * 1 * (1 << 30));
}
