import pytest

from img_creator.prompting import enhance_prompt


def test_prompt_enhancement_keeps_original_text():
    result = enhance_prompt("motor in a factory", True)
    assert result.startswith("motor in a factory")
    assert "photorealistic" in result


def test_prompt_enhancement_can_be_disabled():
    assert enhance_prompt("  motor   in factory ", False) == "motor in factory"


def test_empty_prompt_rejected():
    with pytest.raises(ValueError):
        enhance_prompt("   ", True)
