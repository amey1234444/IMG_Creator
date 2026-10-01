# Detailed image understanding and training captions

This stage improves the descriptions paired with training images. It does not
make the diffusion model memorize every pixel or recover invisible information.
A pretrained local vision-language model inspects an image, proposes structured
evidence, and drafts a caption. The owner corrects the proposal before it can
change approved training data. Vision analysis itself does not train a model.

## What it extracts

| Field | Meaning and limitation |
|---|---|
| Scene | Overall visible scene and main subjects |
| Objects | Up to 24 named objects, visible attributes and approximate normalized boxes |
| Relationships | Relative positions and observable interactions |
| Composition | Framing and visible arrangement; inferred camera specifications are avoided |
| Lighting | Visible light direction and appearance, not measured light parameters |
| Colors and materials | Appearance-based descriptions; material identity can be uncertain |
| Visible text | Proposed transcription with uncertainty labels; not guaranteed OCR |
| Uncertainties | Obscured, ambiguous or unreadable details |
| Suggested caption | Up to 1200 characters, with important subject details first |

The report schema rejects malformed JSON, extra fields, invalid coordinates and
non-finite numbers. This checks structure, not factual accuracy. `clear` and
`uncertain` are model descriptions, **not calibrated confidence scores**.

## Multi-view inspection

The owner selects Overview (one view), Detailed (up to five), or Deep inspection
(up to ten). Every mode includes the whole image. Detailed adds four overlapping
60%-width/height crops. Deep adds nine overlapping 40%-width/height crops. Close-up
crops are used when both source dimensions are at least 768 pixels; smaller images
use the overview. Each view is capped at a 1024-pixel longest edge, then the model
processor enforces its own visual-token budget. The prompt includes source crop
coordinates and asks the model to deduplicate objects and report all boxes in
full-image coordinates. This approach can reveal details hidden by a single
thumbnail, but it does not inspect every original pixel at full resolution.

All views are given to one local inference call so the overview supplies context
for the close-ups. The maximum output is 4096 tokens, with a soft 600-second
generation limit. A soft limit cannot interrupt a stalled accelerator kernel or
model download; use normal process supervision. Dense images can exceed the
object/text/output budgets, and a truncated report fails validation. Try Overview
or prepare focused images if Deep exceeds the worker's memory budget.

The model is `Qwen/Qwen2.5-VL-7B-Instruct`, pinned at
`cc594898137f460bfe9f0759e9844b3ce807cfb5`. The integration disables remote Python
code and loads safetensors. It records the model revision, prompt version and
hash, view hashes and coordinates, actual processor grid shapes, input/output
token counts and elapsed time. This is a deployable starting model, not a claim
that it is the best available or that its extraction is perfect.

## Admin workflow

1. Upload images and assign related photographs to the same subject/shoot group.
2. Choose an inspection depth and click **Analyze image details**. The background
   worker handles it; the web request does not load a model.
3. Click **Refresh analysis**. Inspect objects, visible text, uncertainties and
   the structured report. Open object crop downloads to inspect proposed regions.
4. Correct the JSON observations and copy/edit the suggested caption. Save any
   subject/shoot group edits using the existing asset review controls.
5. Click **Approve corrected analysis and caption** only after checking the source.
   This saves a separate review and approves the selected caption for training.
6. Queue the existing LoRA training flow. It freezes the approved caption and
   links its dataset version and run snapshot to the analysis review. Existing
   group-based validation splitting stays in place.

Object crops are rectangles at the original normalized image resolution. They
are not background removal, precise masks, or independently approved training
samples. Crop downloads use the displayed review's boxes after correction. Do
not upload many crops as independent validation groups: that would leak nearly
identical source content between training and evaluation.

Training uses the **reviewed image-caption pair**, not the full JSON report as an
extra conditioning channel. Reports are retained for inspection and lineage.
Long captions can still be truncated by the diffusion text encoder: put salient
subject attributes first rather than filling the caption with every observation.
Correct captions cannot guarantee learned anatomy, typography or material
physics. Dataset diversity, accurate labels, training resolution, optimization
settings and held-out evaluation still determine generalization.

## Durable schema and failure handling

Schema revision 3 is additive. Run `python -m img_creator.platform.manage init-db`
after deploying this version and before starting workers. It preserves existing
assets, captions, users, generations and training runs.

| Table | Durable record |
|---|---|
| `image_analyses` | Source checksum, model/prompt/mode fingerprint, queue status and proposal |
| `image_analysis_attempts` | Completed attempt output, metrics and errors; retained across retries |
| `image_analysis_reviews` | Owner corrections, chosen caption, reviewer and timestamp |
| `dataset_version_analyses` | Dataset-version/asset association to the exact approved review |

Repeated analysis requests for the same source/model/prompt/mode return the same
job. Failed jobs can be explicitly retried; previous completed attempt records
remain. Workers claim jobs with leases, and expired jobs fail for manual retry.
A worker that loses its lease cannot publish a result. The source checksum is
verified before inference and crop extraction. Successful analysis does **not**
change captions or approval automatically. Manual edits to a caption detach its
review lineage in future snapshots if the text no longer matches that review;
already-frozen snapshots remain unchanged.

Endpoints are owner-only, require the existing session/CSRF controls and share
account API rate limits. Queue/retry requests are limited to 60 per owner per
hour; new submissions also check the 100-active-job queue threshold. Region
downloads are limited to 60 per minute per owner. These are worker cost controls,
not an edge DDoS protection layer. Reports and raw outputs may contain text found
in the original image: protect database backups like the original training data.
Visible image text is treated as untrusted data, never executable instructions.
No report content is executed as code or used as an outbound URL.

## Run the separate vision worker

```bash
# On a suitable model worker, using a compatible Torch/Torchvision pair:
pip install -e '.[platform,vision]'
export VISION_ANALYSIS_ENABLED=true
export VISION_DEVICE=cuda
python -m img_creator.platform.vision_worker
# Or process one queued image and exit:
python -m img_creator.platform.vision_worker --once
```

Set `VISION_ANALYSIS_ENABLED=true` on the web service too, only after provisioning
the worker. Share the private storage and database settings used by the platform.
The vision model is cached once per worker process. It is separate from the
image generator, upscaler and LoRA trainer; avoid competing GPU processes unless
you have measured memory headroom. The default Render CPU web image does not
include these optional model dependencies.

Initial model download requires Hugging Face access and sufficient disk space;
images are processed locally rather than transmitted to an external vision API.
After caching model files, operators can set `HF_HUB_OFFLINE=1` to prevent Hub
network calls. CPU and MPS device choices are available but have not been
benchmarked with this model. The pinned processor/tokenizer was loaded and exercised with five views using
Transformers 5.18.0; no production model weights were downloaded or GPU inference
performed during implementation. The optional vision dependency stays below
Transformers 5.23 until its changed image-sizing API is integrated and verified.

## Acceptance checks

Automated tests verify crop coverage, structured-report validation, model-call
configuration, access control, idempotency, source hash validation, failed-attempt
retention, manual approval and frozen training lineage. Mock predictions do not
establish OCR or recognition accuracy. Before enabling this workflow for a real
collection, compare reports to manually annotated held-out images. Measure missed
objects, invented details, box overlap, text character/word errors and owner
correction time. Include fine patterns, reflections, low light, multilingual
text and occlusion. Do not advertise perfect extraction or learning of everything
in an image.

Reference: [official pinned Qwen model card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/cc594898137f460bfe9f0759e9844b3ce807cfb5/README.md).

Recheck the real processor contract without loading model weights with:
`python tests/vision_processor_smoke.py`. This downloads only processor/tokenizer
files on its first run and is also exercised by the vision-processor CI job.
