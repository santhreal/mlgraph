//! ViT-Tiny bandwidth analysis: before and after fusion.
//!
//! Run with: `cargo run --example vit_optimize`

use mlgraph::analysis::bandwidth::BandwidthAnalysis;
use mlgraph::fusion::attn_block::AttentionBlockFusion;
use mlgraph::fusion::ffn_block::FfnBlockFusion;
use mlgraph::models::vit::{build_vit, ViTConfig};
use mlgraph::pass::{AnalysisPass, TransformPass};

fn main() {
    // ── Build the ViT-Tiny graph ────────────────────────────────────────

    let config = ViTConfig::tiny();
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  mlgraph  -  Bandwidth-Aware Computation Graph Optimizer    ║");
    println!("║  Model: {} (inference, {})               ║", config.name, config.dtype);
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();

    println!("Configuration:");
    println!("  Image size:    {} × {}", config.image_size, config.image_size);
    println!("  Patch size:    {}", config.patch_size);
    println!("  Hidden dim:    {}", config.hidden_dim);
    println!("  Heads:         {}", config.num_heads);
    println!("  Head dim:      {}", config.head_dim());
    println!("  MLP dim:       {}", config.mlp_dim);
    println!("  Layers:        {}", config.num_layers);
    println!("  Seq length:    {} (patches + CLS)", config.seq_len());
    println!("  Batch size:    {}", config.batch_size);
    println!();

    let graph = build_vit(&config).expect("failed to build ViT graph");
    println!(
        "Graph constructed: {} nodes, {} tensors",
        graph.num_nodes(),
        graph.num_tensors()
    );
    println!();

    // ── Analyze BEFORE fusion ───────────────────────────────────────────

    println!("═══════════════════════════════════════════════════════════");
    println!("  BEFORE FUSION (unfused graph)");
    println!("═══════════════════════════════════════════════════════════");

    let before_report = BandwidthAnalysis
        .analyze(&graph)
        .expect("analysis failed");

    // Print per-layer summary (first layer only, to keep output readable).
    println!();
    println!("Per-layer breakdown (Layer 0):");
    let layer0_nodes: Vec<_> = before_report
        .nodes
        .iter()
        .filter(|n| n.node_name.starts_with("L0."))
        .collect();
    print_node_table(&layer0_nodes);

    let per_layer_read: u64 = layer0_nodes.iter().map(|n| n.hbm_read).sum();
    let per_layer_write: u64 = layer0_nodes.iter().map(|n| n.hbm_write).sum();
    let per_layer_total = per_layer_read + per_layer_write;
    println!();
    println!(
        "  Per-layer HBM traffic:  {} (read: {}, write: {})",
        format_bytes(per_layer_total),
        format_bytes(per_layer_read),
        format_bytes(per_layer_write)
    );
    println!(
        "  {:>2} layers total:        {}",
        config.num_layers,
        format_bytes(per_layer_total * config.num_layers as u64)
    );

    println!();
    println!(
        "  Graph total HBM traffic: {}",
        format_bytes(before_report.total_hbm_traffic)
    );
    println!(
        "  Graph total FLOPs:       {}",
        format_flops(before_report.total_flops)
    );
    println!(
        "  Arithmetic intensity:    {:.2} FLOP/byte ({})",
        before_report.arithmetic_intensity,
        classify(before_report.arithmetic_intensity)
    );

    // ── Apply fusion passes ─────────────────────────────────────────────

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  APPLYING FUSION PASSES");
    println!("═══════════════════════════════════════════════════════════");

    let mut fused_graph = build_vit(&config).expect("failed to rebuild graph");

    let attn_fused = AttentionBlockFusion
        .transform(&mut fused_graph)
        .expect("attention fusion failed");
    println!("  Attention blocks fused: {attn_fused}");

    let ffn_fused = FfnBlockFusion
        .transform(&mut fused_graph)
        .expect("FFN fusion failed");
    println!("  FFN blocks fused:       {ffn_fused}");

    println!(
        "  Graph after fusion:     {} nodes, {} tensors",
        fused_graph.num_nodes(),
        fused_graph.num_tensors()
    );

    // ── Analyze AFTER fusion ────────────────────────────────────────────

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  AFTER FUSION");
    println!("═══════════════════════════════════════════════════════════");

    let after_report = BandwidthAnalysis
        .analyze(&fused_graph)
        .expect("analysis failed");

    println!();
    let fused_nodes: Vec<_> = after_report.nodes.iter().collect();
    print_node_table(&fused_nodes);

    println!();
    println!(
        "  Graph total HBM traffic: {}",
        format_bytes(after_report.total_hbm_traffic)
    );
    println!(
        "  Graph total FLOPs:       {}",
        format_flops(after_report.total_flops)
    );
    println!(
        "  Arithmetic intensity:    {:.2} FLOP/byte ({})",
        after_report.arithmetic_intensity,
        classify(after_report.arithmetic_intensity)
    );

    // ── Comparison ──────────────────────────────────────────────────────

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  COMPARISON");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    let reduction = if after_report.total_hbm_traffic > 0 {
        before_report.total_hbm_traffic as f64 / after_report.total_hbm_traffic as f64
    } else {
        0.0
    };

    println!(
        "  HBM traffic before:      {}",
        format_bytes(before_report.total_hbm_traffic)
    );
    println!(
        "  HBM traffic after:       {}",
        format_bytes(after_report.total_hbm_traffic)
    );
    println!("  ─────────────────────────────────────");
    println!("  Bandwidth reduction:     {reduction:.1}×");
    println!();
    println!(
        "  Nodes before:            {}",
        before_report.nodes.len()
    );
    println!(
        "  Nodes after:             {}",
        after_report.nodes.len()
    );
    println!();

    // Rough speedup estimate for bandwidth-bound models.
    let estimated_speedup = reduction.min(5.0); // cap at 5× (kernel launch savings)
    println!(
        "  Estimated speedup:       {estimated_speedup:.1}× \
         (bandwidth-limited regime, seq_len={})",
        config.seq_len()
    );
    println!();
    println!("  Note: Actual speedup depends on GPU architecture, memory");
    println!("  hierarchy, and kernel implementation. These numbers represent");
    println!("  the theoretical HBM bandwidth reduction from fusion.");
}

fn print_node_table(nodes: &[&mlgraph::analysis::bandwidth::NodeBandwidth]) {
    println!(
        "  {:<28} {:>10} {:>10} {:>12}",
        "Operation", "HBM Read", "HBM Write", "HBM Total"
    );
    println!("  {}", "─".repeat(64));
    for node in nodes {
        println!(
            "  {:<28} {:>10} {:>10} {:>12}",
            truncate(&node.node_name, 28),
            format_bytes(node.hbm_read),
            format_bytes(node.hbm_write),
            format_bytes(node.hbm_total),
        );
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}…", &s[..max_len - 1])
    }
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

fn format_flops(flops: u64) -> String {
    if flops >= 1_000_000_000_000 {
        format!("{:.2} TFLOPS", flops as f64 / 1e12)
    } else if flops >= 1_000_000_000 {
        format!("{:.2} GFLOPS", flops as f64 / 1e9)
    } else if flops >= 1_000_000 {
        format!("{:.2} MFLOPS", flops as f64 / 1e6)
    } else {
        format!("{flops} FLOPS")
    }
}

fn classify(intensity: f64) -> &'static str {
    if intensity < 1.0 {
        "bandwidth-bound"
    } else if intensity < 10.0 {
        "transitional"
    } else {
        "compute-bound"
    }
}
