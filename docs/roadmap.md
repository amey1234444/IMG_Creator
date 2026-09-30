# Roadmap

## V0.2 — implemented, hardware acceptance pending

Local studio and API, model profiles, learned-SR adapter, local LoRA registry, reference conditioning, reproducible image exports, disk-backed dataset preparation, tar shards, official trainer launcher, and fixed benchmark prompts.

## Next acceptance gates

1. Generate a native image on the intended CUDA/MPS hardware and record peak memory and timing.
2. Test one trusted Real-ESRGAN/SwinIR checkpoint, compare native/Lanczos/learned output and inspect tile edges.
3. Run a 10-step LoRA smoke test, resume from a checkpoint, reload the adapter and compare fixed prompts.
4. Profile preparation/export on a real 1,000-image sample; verify train/validation groups and storage estimates.

## Later implementation

- Streaming bucketed training with resumable shuffling and multi-GPU coverage tests.
- Disk-backed embedding/latent caches with invalidation rules.
- Perceptual near-duplicate review and caption-quality review tools.
- Durable jobs, progress, cancellation, retention and recovery.
- Dedicated image-editing API with bounded uploads.
- Tiled diffusion refinement, benchmark grids and optional objective metrics.
- Quantization profiles measured on actual hardware.

There are no claims of perfected anatomy, unrestricted output, universal prompt adherence, or trained model quality.
