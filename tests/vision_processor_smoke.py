"""Opt-in real processor smoke check; downloads tokenizer/config, never model weights.

Run: python tests/vision_processor_smoke.py
Requires the vision extra and network access on the first run.
"""

from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from img_creator.platform.vision import MODEL_ID, MODEL_REVISION, image_views, report_prompt

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
    trust_remote_code=False,
    min_pixels=4 * 28 * 28,
    max_pixels=768 * 28 * 28,
)
views, manifest = image_views(Image.new("RGB", (800, 800), "blue"), "detail")
text = processor.apply_chat_template(
    [
        {
            "role": "user",
            "content": [{"type": "image"} for _ in views]
            + [{"type": "text", "text": report_prompt(manifest, (800, 800))}],
        }
    ],
    tokenize=False,
    add_generation_prompt=True,
)
inputs = processor(text=[text], images=views, padding=True, return_tensors="pt")
assert inputs.image_grid_thw.shape[0] == 5
assert inputs.input_ids.shape[0] == 1
assert Qwen2_5_VLForConditionalGeneration.__name__ == "Qwen2_5_VLForConditionalGeneration"
print(
    {
        "views": len(views),
        "input_shape": list(inputs.input_ids.shape),
        "image_grid_thw": inputs.image_grid_thw.tolist(),
        "revision": MODEL_REVISION,
    }
)
