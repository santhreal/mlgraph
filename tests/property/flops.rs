use mlgraph::op::Op;
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_matmul_flops_invariants(
        m in 1..=1000usize,
        n in 1..=1000usize,
        k in 1..=1000usize,
        b in 1..=10usize,
    ) {
        let op = Op::MatMul;
        let shape_a = vec![b, m, k];
        let shape_b = vec![k, n];
        let flops = op.flops(&[&shape_a, &shape_b], &[&[b, m, n]]);
        
        // Flops should be exactly 2 * B * M * N * K
        let expected = 2 * (b as u64) * (m as u64) * (n as u64) * (k as u64);
        prop_assert_eq!(flops, expected);
        prop_assert!(flops > 0);
    }

    #[test]
    fn test_linear_flops_invariants(
        batch in 1..=100usize,
        seq in 1..=100usize,
        in_features in 1..=500usize,
        out_features in 1..=500usize,
        bias in any::<bool>(),
    ) {
        let op = Op::Linear { out_features, bias };
        let shape = vec![batch, seq, in_features];
        let flops = op.flops(&[&shape], &[&[batch, seq, out_features]]);
        
        let batch_seq = (batch as u64) * (seq as u64);
        let expected = 2 * batch_seq * (in_features as u64) * (out_features as u64) +
                       if bias { batch_seq * (out_features as u64) } else { 0 };
                       
        prop_assert_eq!(flops, expected);
        prop_assert!(flops > 0);
    }

    #[test]
    fn test_activation_flops_monotonicity(
        elements in 1..=10000usize,
    ) {
        let shape = vec![elements];
        
        let relu = Op::Relu;
        let relu_flops = relu.flops(&[&shape], &[&shape]);
        
        let gelu = Op::Gelu;
        let gelu_flops = gelu.flops(&[&shape], &[&shape]);
        
        let silu = Op::Silu;
        let silu_flops = silu.flops(&[&shape], &[&shape]);

        // Non-negativity
        prop_assert!(relu_flops > 0);
        prop_assert!(gelu_flops > 0);
        prop_assert!(silu_flops > 0);
        
        // Exact proportionality checks based on implementation
        prop_assert_eq!(relu_flops, elements as u64);
        prop_assert_eq!(gelu_flops, 8 * elements as u64);
        prop_assert_eq!(silu_flops, 8 * elements as u64);
    }
}
