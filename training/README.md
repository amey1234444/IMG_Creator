# Dataset and LoRA training guide

The practical first goal is a measurable improvement in one domain, such as industrial equipment, product photography or a consistent illustration style. The base model already supplies broad visual knowledge. Training a 4B model from scratch is outside this project.

## 1. Build a versioned dataset

Use images you own or are authorized to train on. Keep originals in stable storage. Supply `file_name`, an accurate `text` caption, `source`, `license`, and optionally `group_id` in JSONL. Describe visible subjects, counts, relationships, viewpoint, materials and lighting. Do not fill every caption with vague quality keywords.

Use the commands in the main README to prepare the SQLite inventory. Review `rejected.jsonl`, `report.json` and the `review_flags` column before exporting. Low edge variance is a heuristic, not a reliable blur classifier; flat illustrations may be flagged. Watermarks, incorrect captions, semantic near-duplicates, provenance and harmful content require review or a separate audited classifier. No automatic training corpus is downloaded by this project.

Exact duplicate removal happens before splitting. Related captures must share a `group_id`. A deterministic hash assigns approximately 5% of groups to validation. Small datasets can have no validation groups; inspect the counts and use more independent groups or a larger validation fraction. There is no forced one-image holdout that would split a related group.

Buckets use width×height: 1024×1024, 1216×832, 832×1216, 1344×768, 768×1344, 1152×896 and 896×1152. Upstream CLI lists are **height,width**, so the config converts that order. Training resizes proportionally and crops instead of stretching.

## 2. Storage and scale

Preparation decodes one image at a time and keeps hashes and rows on disk. Runtime grows with total bytes decoded; this is not a parallel distributed ingestion service. A completed report marks a finished preparation run. If interrupted, keep the original input and rebuild into a new output directory; preparation does not yet resume an incomplete index.

Export reads rows sequentially, verifies each original file hash, and emits tar shards of at most 1,000 samples by default. It also rolls shards near 1 GB using `--shard-bytes`; a single image and tar overhead can exceed that target. Shards contain `<key>.png`, `<key>.txt`, `<key>.json`. PNG normalization preserves decoded pixels and strips EXIF/GPS, but can take substantially more disk than original JPEGs. Originals are not modified. A failed export removes its newly created destination; rerun into a new version after resolving the error.

Illustrative capacity arithmetic: one million originals averaging 1 MB require about 1 TB before indexes, normalized shards, validation copies or checkpoints. Measure actual export size on 1,000 images before budgeting storage. Keep image payloads in object storage/shared disk and the versioned manifest alongside them; do not put them in Git.

## 3. Install the pilot trainer

Use Linux/CUDA for the initial training pilot. The example configuration uses BF16; check your GPU supports it and choose FP16 if appropriate. Apple MPS training and multi-GPU launch have not been validated here.

```bash
pip install -e '.[inference,training]'
git clone https://github.com/huggingface/diffusers.git third_party/diffusers
git -C third_party/diffusers checkout 031b2798addadd1652db7cfba50eacc1079245cf
pip install -e third_party/diffusers
accelerate config
```

Use a matching torch/torchvision pair. Optional upstream optimizers such as bitsandbytes and Prodigy are not enabled by this project's configuration. No remote text encoder, W&B account, or automatic model upload is required.

Export 50–500 images initially:

```bash
python -m training.export_dataset \
  --index prepared/v1/index.sqlite3 \
  --output exports/pilot-v1 --split train --pilot-limit 500
```

Print and validate the launch command:

```bash
python -m training.train_lora \
  --diffusers-checkout third_party/diffusers \
  --dataset exports/pilot-v1 \
  --output checkpoints/industrial-v1
```

Append `--run` to start. For recovery, append `--resume latest --run`. Run from the repository root so the default configuration resolves correctly. The launcher validates the exact checkout commit, actual upstream flags, local caption count and base-model selection. It invokes an argument vector, never a shell string.

The official trainer freezes base weights and adds LoRA adapters. This wrapper delegates optimization to that trainer; it does not implement a second, unverified flow-matching loss. The adapter output is `pytorch_lora_weights.safetensors`. Register it for `black-forest-labs/FLUX.2-klein-base-4B`, select the Quality profile, and evaluate.

## 4. Initial settings and checks

| Setting | Initial value | Purpose |
|---|---:|---|
| Rank / alpha | 16 / 16 | Small adapter capacity |
| Learning rate | 0.0001 | Starting experiment, tune from validation |
| Batch / accumulation | 1 / 4 | Effective batch 4 on one device |
| Steps | 1,000 | Short pilot, not a universally optimal duration |
| Checkpoints | Every 250, keep 3 | Recover and compare |
| Resolution | Approximately 1 MP buckets | Preserve aspect ratios |
| Precision | BF16 | Requires compatible hardware |
| Gradient checkpointing | Enabled | Trade speed for activation memory |

First run a very short trial by lowering `max_train_steps` in a copied config. Measure peak VRAM, seconds per optimizer step, dataset RAM, output artifacts and validation generation. Estimate full duration as measured seconds/step × planned steps plus validation/checkpoint overhead. Do not infer GPU costs without hardware and provider prices.

For 500 images, batch 1 × accumulation 4 × 1,000 steps is roughly 4,000 sample exposures, or eight dataset passes on one device. Larger datasets need an explicit exposure/epoch plan; a fixed 1,000-step run barely touches a million-image collection.

## 5. Evaluate before scaling

Run the fixed prompt suite on the untouched base model and each candidate adapter using the same profile, seed, resolution and upscaler. Include domain and unrelated prompts to detect regressions. Review prompt adherence, object count, relationships, material appearance, anatomy and artifacts. Check held-out images for memorization or identity leakage. Keep the best checkpoint from evaluation, not automatically the latest one.

The included benchmark runner generates outputs and a scoring CSV. It does not calculate FID, CLIP or aesthetic metrics. Do not claim a win until the images have actually been reviewed.

## 6. Scaling beyond the pilot

The official example materializes image tensors, so the launcher caps its input at 2,000 images. **The exported million-image tar collection must not be fed into this pilot launcher.** A separate streaming trainer integration is the next implementation milestone.

For that integration, use the exported shards with WebDataset or an equivalent streaming loader. Implement deterministic shard ordering, per-worker/rank sharding, epoch state and resumable sampler state; retain group-isolated validation. Profile decode throughput before adding distributed training. Cache latents/text embeddings on disk with a key incorporating file hash, caption hash, model revision, bucket and crop policy; avoid unbounded RAM caches. Training batches must contain the same bucket dimensions. Checkpoint optimizer, scheduler, RNG, sampler position and adapter together. Compare a resumed run with an uninterrupted short run before running a large job.

The app has no training HTTP endpoint and will not start GPU spending from a generation request. Dataset collection, training and model promotion remain explicit CLI operations.
