use mlgraph::error::Error;
use mlgraph::op::Op;

#[test]
fn test_transpose_empty_inputs() {
    let op = Op::Transpose { perm: vec![] };
    let result = op.infer_shapes(&[]);
    assert!(matches!(result, Err(Error::ShapeMismatch { .. })));
}

#[test]
fn test_transpose_permutation_too_short() {
    let op = Op::Transpose { perm: vec![0] };
    let shape = vec![10, 20];
    let result = op.infer_shapes(&[&shape]);
    assert!(matches!(result, Err(Error::ShapeMismatch { .. })));
}

#[test]
fn test_transpose_permutation_too_long() {
    let op = Op::Transpose { perm: vec![0, 1, 2] };
    let shape = vec![10, 20];
    let result = op.infer_shapes(&[&shape]);
    assert!(matches!(result, Err(Error::ShapeMismatch { .. })));
}

#[test]
fn test_transpose_permutation_out_of_bounds() {
    let op = Op::Transpose { perm: vec![0, 2] };
    let shape = vec![10, 20];
    let result = op.infer_shapes(&[&shape]);
    assert!(matches!(result, Err(Error::ShapeMismatch { .. })));
}

#[test]
fn test_transpose_permutation_duplicate_indices() {
    let op = Op::Transpose { perm: vec![0, 0] };
    let shape = vec![10, 20];
    let result = op.infer_shapes(&[&shape]);
    assert!(matches!(result, Err(Error::ShapeMismatch { .. })));
}

#[test]
fn test_transpose_zero_dims() {
    let op = Op::Transpose { perm: vec![1, 0] };
    let shape = vec![0, 10];
    let result = op.infer_shapes(&[&shape]).unwrap();
    assert_eq!(result[0], vec![10, 0]);
}

#[test]
fn test_transpose_extreme_dims() {
    let op = Op::Transpose { perm: vec![1, 0] };
    let shape = vec![usize::MAX, usize::MIN];
    let result = op.infer_shapes(&[&shape]).unwrap();
    assert_eq!(result[0], vec![usize::MIN, usize::MAX]);
}

#[test]
fn test_transpose_large_rank() {
    let rank = 1000;
    let mut perm: Vec<usize> = (0..rank).collect();
    perm.reverse();
    let op = Op::Transpose { perm: perm.clone() };
    
    let shape: Vec<usize> = (0..rank).collect();
    let result = op.infer_shapes(&[&shape]).unwrap();
    
    let mut expected = shape.clone();
    expected.reverse();
    assert_eq!(result[0], expected);
}

#[test]
fn test_transpose_duplicate_at_tail_of_large_perm_still_rejected() {
    // Regression lock for the O(n) seen-set duplicate check (previously an
    // O(n^2) per-index rescan): a duplicate that only appears at the END of a
    // large permutation must still be caught. A seen-set that stops early or
    // mis-indexes would accept it and produce a silently wrong output shape.
    let rank = 512;
    let mut perm: Vec<usize> = (0..rank).collect();
    perm[rank - 1] = 0; // duplicate of index 0 at the tail
    let op = Op::Transpose { perm };
    let shape: Vec<usize> = (1..=rank).collect();
    let result = op.infer_shapes(&[&shape]);
    assert!(
        matches!(result, Err(Error::ShapeMismatch { .. })),
        "a tail duplicate must be rejected, got {result:?}"
    );
}
