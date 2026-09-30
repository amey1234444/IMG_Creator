# Architecture and operations

`schemas/generation.py` validates all API and UI requests. `presets.py` separates generation dimensions from export sizes and selects the model profile. `generator.py` owns one backend and a nonblocking lock that covers loading, adapter mutation, inference, upscaling and artifact persistence. Concurrent API generation/upscale calls return 429; Gradio serializes work through its queue.

`pipelines/flux_pipeline.py` imports PyTorch/Diffusers lazily. It switches models by releasing the previous pipeline, uses supported CUDA BF16 or FP16, and uses conservative FP32 on MPS/CPU. CUDA model CPU offload is configurable. VAE tiling reduces decode pressure. Adapters are registered locally and must match the model ID; request data cannot choose filesystem locations.

`upscale.py` owns the optional learned backend. Importing the API does not require PyTorch. `metadata.py` stages an image and JSON in a temporary directory, then renames the directory on the same filesystem. Failed writes remove staging data. An interrupted process can leave a hidden staging directory; it is never a valid published artifact and can be removed after confirming no process owns it.

Artifacts use 32-hex IDs. File endpoints reject invalid IDs and symlinked artifact paths. This is a local, single-owner service, not a tenant-isolated public platform. Optional API keys cover generation and artifact reads; Gradio binds to localhost without a public share link. No account database, billing, distributed rate limiter, durable queue or multi-tenant authorization is implemented.

A batch saves each finished image separately. If a later image fails, earlier completed artifacts remain on disk, but the request fails as a whole; there is no batch-resume endpoint. Retention/deletion is operator-managed. The API does not accept arbitrary uploaded filesystem paths. Reference-image conditioning is currently a UI/Python feature.

`training/` runs from the repository root and is intentionally separate from the inference wheel. It writes new dataset versions without overwriting an existing destination. SQLite avoids holding all metadata in RAM; image payloads remain on disk. Exported shards are prepared for a future streaming trainer. The included upstream pilot trainer is explicitly bounded because its internal loader is eager.

Public serving requires a separate deployment decision and capacity tests. Run one process per GPU; multiple uvicorn workers duplicate model memory and do not coordinate this process-local lock.
