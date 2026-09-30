from __future__ import annotations
import logging
import secrets
import gradio as gr
from .generator import ImageGenerator
from .presets import PRESETS
from .prompting import STYLES
from .schemas import GenerateRequest

log = logging.getLogger(__name__)


def build_ui(service=None):
    service = service or ImageGenerator()

    def run(
        prompt,
        aspect,
        quality,
        resolution,
        seed,
        enhance,
        style,
        count,
        steps,
        guidance,
        upscaler,
        fmt,
        lora,
        weight,
        reference,
    ):
        try:
            request = GenerateRequest(
                prompt=prompt,
                aspect_ratio=aspect,
                quality=quality,
                output_resolution=resolution,
                seed=None if seed in (-1, None) else int(seed),
                prompt_enhancement=enhance,
                style=style,
                num_images=int(count),
                steps=None if steps == 0 else int(steps),
                guidance_scale=None if guidance == -1 else guidance,
                upscaler=upscaler,
                output_format=fmt,
                lora=lora or None,
                lora_weight=weight,
            )
            if reference is not None and reference.width * reference.height > 16_000_000:
                raise ValueError("Reference image must be at most 16 megapixels")
            items = service.generate_batch(request, reference=reference)
            paths = [str(service.store.image_path(m["generation_id"])) for m in items]
            downloads = paths + [str(service.store.directory(m["generation_id"]) / "metadata.json") for m in items]
            return (
                [(p, f"Seed {m['seed']} · {m['final_width']} × {m['final_height']}") for p, m in zip(paths, items)],
                items,
                downloads,
            )
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        except Exception as exc:
            log.exception("Generation failed")
            raise gr.Error(
                "Generation failed. Check the terminal for model access, installation or memory errors."
            ) from exc

    with gr.Blocks(title="IMG Creator") as demo:
        gr.Markdown("# IMG Creator\nCreate locally. Shape the composition, choose the finish, keep every setting.")
        with gr.Row():
            with gr.Column(scale=5):
                prompt = gr.Textbox(
                    label="Describe your image",
                    lines=5,
                    placeholder="A brushed-steel motor in a sunlit factory, documentary photograph...",
                )
                with gr.Row():
                    aspect = gr.Dropdown(list(PRESETS), value="1:1", label="Aspect ratio")
                    resolution = gr.Dropdown(list(PRESETS["1:1"]), value="Native", label="Export size")
                quality = gr.Radio(
                    ["fast", "quality", "ultra"],
                    value="fast",
                    label="Generation profile",
                    info="Fast: distilled model. Quality: base model. Ultra: base model + learned super-resolution.",
                )
                with gr.Row():
                    style = gr.Dropdown(list(STYLES), value="neutral", label="Style for prompt enhancement")
                    enhance = gr.Checkbox(value=False, label="Enhance my prompt")
                with gr.Accordion("Reference image / editing", open=False):
                    reference = gr.Image(
                        type="pil", image_mode="RGB", sources=["upload"], label="Upload an optional reference image"
                    )
                with gr.Accordion("Advanced controls", open=False):
                    with gr.Row():
                        seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                        randomize = gr.Button("Random seed")
                    count = gr.Slider(1, 4, value=1, step=1, label="Number of images")
                    steps = gr.Slider(0, 100, value=0, step=1, label="Steps (0 = profile default)")
                    guidance = gr.Slider(-1, 20, value=-1, step=0.1, label="Guidance (-1 = profile default)")
                    upscaler = gr.Dropdown(
                        ["lanczos", "learned"],
                        value="lanczos",
                        label="Upscaler",
                        info="Lanczos changes dimensions. Learned requires local SR weights.",
                    )
                    lora = gr.Dropdown(
                        [""] + [v["id"] for v in service.registry.list()], value="", label="LoRA adapter"
                    )
                    weight = gr.Slider(0, 2, value=1, step=0.05, label="LoRA strength")
                    fmt = gr.Radio(["png", "jpeg", "webp"], value="png", label="File format")
                generate = gr.Button("Generate images", variant="primary")
            with gr.Column(scale=7):
                gallery = gr.Gallery(label="Your images", columns=2, object_fit="contain", height=580)
                downloads = gr.File(label="Download images and settings", file_count="multiple")
                with gr.Accordion("Generation settings", open=False):
                    metadata = gr.JSON()
        aspect.change(lambda a: gr.Dropdown(choices=list(PRESETS[a]), value="Native"), aspect, resolution)
        quality.change(lambda q: gr.Dropdown(value="learned" if q == "ultra" else "lanczos"), quality, upscaler)
        randomize.click(lambda: secrets.randbits(32), outputs=seed)
        generate.click(
            run,
            [
                prompt,
                aspect,
                quality,
                resolution,
                seed,
                enhance,
                style,
                count,
                steps,
                guidance,
                upscaler,
                fmt,
                lora,
                weight,
                reference,
            ],
            [gallery, metadata, downloads],
            concurrency_limit=1,
        )
    return demo.queue(max_size=8, default_concurrency_limit=1)


def main():
    build_ui().launch(server_name="127.0.0.1", share=False)
