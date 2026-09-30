# High-resolution processing

Base diffusion runs near one megapixel. Fast uses the distilled Klein model; Quality and Ultra use the base model. No second diffusion refinement pass is currently implemented.

Lanczos export uses an aspect-preserving center crop plus resize. Its metadata reads `lanczos-resize` or `native`. It provides requested dimensions without learned detail reconstruction.

Learned SR loads a local, trusted RGB 2×/4× model through Spandrel. Real-ESRGAN and SwinIR are candidate weight families. Each input tile has 32 pixels of extra context. Only its central output is pasted into the host-memory canvas, avoiding duplicate coverage. This reduces boundary context loss but does not guarantee seam-free results for every architecture; inspect repeated patterns and tile boundaries using your selected weights. SwinIR-style models can have padding/window requirements, handled by their descriptor.

The default core tile is 256×256. Lower it to reduce accelerator memory; raise it if measured throughput improves and memory permits. FLUX and SR weights may coexist in device memory, especially when CPU offload is disabled. CPU offload, native output and a smaller tile are the first memory-reduction options. An OOM is surfaced as an error; no silent quality downgrade occurs.

A 1344×768 image with 4× SR produces 5376×3072 before final cropping/resampling. A 3840×2160 export downsizes that result; a 7680×4320 export further resizes it. Both operations are recorded. They should not be described as native 4K/8K diffusion. Learned intermediates above 64 megapixels are rejected; re-upscale from the original generation instead of repeatedly enlarging an export.

Weights, downloads, architecture compatibility and actual visual quality require local validation. The test suite verifies tile coverage, crop behavior, missing-weight errors and metadata without downloading SR weights. True tiled diffusion refinement and automatic image-quality metrics remain planned.
