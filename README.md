# mlgraph

![Status: alpha](https://img.shields.io/badge/status-alpha-blue.svg)

Bandwidth-aware computation graph optimizer for ML inference - fuses transformer layers to minimize HBM traffic.

## What it does

`mlgraph` analyzes machine learning model computation graphs to optimize High Bandwidth Memory (HBM) utilization during inference. It identifies fusion candidates across transformer layers (attention heads, FFN blocks, activation functions) to minimize intermediate memory reads and writes.

Key features:
- **Bandwidth analysis**: Calculates exact HBM read/write byte traffic and arithmetic intensity.
- **Layer fusion**: Merges adjacent nodes into fused megakernels to eliminate SRAM-to-HBM spilling.
- **Model graphs**: Pre-built representations for vision transformers (ViT) and LLM blocks.

## Quick start

```rust
use mlgraph::models::vit::{build_vit, ViTConfig};
use mlgraph::analysis::bandwidth::BandwidthAnalysis;
use mlgraph::pass::AnalysisPass;

let graph = build_vit(&ViTConfig::tiny()).expect("failed to build ViT graph");
let report = BandwidthAnalysis.analyze(&graph).expect("failed to analyze bandwidth");
println!("Total HBM traffic: {} bytes", report.total_hbm_traffic);
```

## When to use / when not

### When to use
- Graph optimization and kernel fusion passes targeting GPU memory bottlenecks.
- HBM traffic estimation for deep learning inference workloads.
- Benchmarking transformer layer fusion strategies prior to codegen.

### When not to use
- Full tensor runtime execution or device buffer allocation (use Vyre or framework runtime engines).

## Compared to alternatives

- **Static graph compilers**: Rely on fixed rewrite rules without quantitative HBM byte-traffic analysis. `mlgraph` measures arithmetic intensity and memory traffic per node.
- **Manual kernel fusion**: Time-consuming and fragile. `mlgraph` automatically identifies multi-op fusion boundaries across computation graphs.

## How it fits in Santh

`mlgraph` lives in `libs/general/` and provides graph optimization and bandwidth profiling for compiler and acceleration pipelines across the Santh ecosystem (such as Vyre and SurgeC).

## Contributing

Contributions require unit, adversarial, property, and gap test coverage. Run `cargo test -p mlgraph` to verify changes.

## License

Licensed under the MIT License ([LICENSE](LICENSE)) or Apache License 2.0.
