from unittest.mock import patch
import pytest
from training.train_lora import command, DIFFUSERS_REVISION


def test_launcher_is_argument_vector_and_validates_revision(tmp_path):
    checkout = tmp_path / "diffusers"
    script = checkout / "examples/dreambooth/train_dreambooth_lora_flux2_klein.py"
    script.parent.mkdir(parents=True)
    flags = [
        "pretrained_model_name_or_path",
        "max_train_steps",
        "dataset_name",
        "image_column",
        "caption_column",
        "output_dir",
        "report_to",
    ]
    script.write_text("\n".join(repr("--" + flag) for flag in flags))
    data = tmp_path / "dataset with spaces"
    data.mkdir()
    (data / "metadata.jsonl").write_text("{}\n")
    cfg = {"pretrained_model_name_or_path": "black-forest-labs/FLUX.2-klein-base-4B", "max_train_steps": 10}
    with patch("subprocess.check_output", return_value=DIFFUSERS_REVISION + "\n"):
        cmd = command(cfg, checkout, data, tmp_path / "output")
        assert str(data.resolve()) in cmd
        assert cmd[1:3] == ["-m", "accelerate.commands.launch"]
        with pytest.raises(ValueError, match="Unsupported"):
            command({**cfg, "push_to_hub": True}, checkout, data, tmp_path / "out")
    with patch("subprocess.check_output", return_value="wrong"):
        with pytest.raises(ValueError, match="pinned"):
            command(cfg, checkout, data, tmp_path / "out")
