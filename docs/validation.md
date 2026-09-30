# Validation record — 30 September 2026

## Executed

- Starter baseline: 5 tests passed before implementation changes.
- Final suite: **39 tests passed** on Python 3.12.14.
- Ruff lint: passed.
- Ruff formatting: 33 Python files checked, passed.
- `compileall` for application, training and evaluation modules: passed.
- Editable package build and installation with `--no-deps --no-build-isolation`: passed.
- Gradio construction: 36 components and 4 callbacks. The generate callback saved a synthetic image and metadata successfully.
- FastAPI roundtrip with a fake model: generation, image retrieval, metadata retrieval and subsequent upscale passed. Authentication, request errors and server error redaction passed.
- API import: no `torch` or `diffusers` module imported; no model weights loaded.
- All configured trainer flags were checked against the retrieved pinned upstream training source.
- Dataset fixtures exercised corrupt/missing/tiny/path-escaping images, byte/pixel duplicates, grouped splits, tar export, pilot export and changed-source detection.
- Concurrency, adapter failure cleanup, seed progression/wraparound, actual output dimensions and failed-save cleanup were tested.

Tested lightweight dependencies: Gradio 6.29.0, FastAPI 0.142.1, Pydantic 2.13.5, Pillow 12.3.0, pytest 9.1.1 and Ruff 0.16.9. One Starlette warning recommends migrating its test client from httpx to httpx2; tests still pass.

The restricted test environment's socket permissions blocked an initial in-process TestClient run. The same tests passed when executed with the necessary socket permissions. This was also reproduced with a minimal independent FastAPI application.

## Not executed

- Actual FLUX weight download or model inference on CUDA, Apple MPS or CPU.
- Actual LoRA optimization, checkpoint recovery or trained-adapter image quality evaluation.
- Learned SR using real Real-ESRGAN/SwinIR weights; tile geometry is tested, visual quality is not.
- Million-image ingestion/export benchmark or multi-GPU/distributed training.
- Pixel-level browser screenshot review of Gradio.
- Hosted CI runs (workflow supplied; check its status on GitHub).

Accordingly, mocked workflow tests establish application behavior, not model quality or end-to-end hardware compatibility. No trained model or dataset is bundled.


## Hosted platform 0.3 validation

Local Python 3.12: 62 tests passed (23 platform checks plus 39 existing checks),
with no model weights or paid provider calls. Ruff lint/format, Python compile,
JavaScript syntax and YAML parsing passed. One upstream Starlette/httpx
compatibility deprecation warning remains. Browser startup in the execution
sandbox was blocked by socket permissions; separate GitHub Actions browser and
Postgres jobs now exercise those environments and publish browser screenshots.
Check the actual workflow results before treating those gates as passed.

Platform coverage includes CSRF/owner enforcement, tenant isolation, concurrent
credit debit, retries, worker leases, unknown provider outcomes, refunds, signed
webhook rejection, payment replay, checkout reuse, invoice-price attribution,
caption review, immutable training snapshots, adapter promotion, upload limits,
provider request contracts and usage aggregation with missing token reports.

No GPU generation, training, live Stripe checkout, S3 integration, load test,
provider-cost audit or photorealism benchmark was run. See `docs/platform.md` for
launch requirements and the distinction between credits, tokens and worker time.
