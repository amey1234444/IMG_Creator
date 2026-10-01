import json
import pytest
from PIL import Image
from img_creator.platform.vision import image_views, parse_report, ImageReport, report_prompt


def report_example():
    return {
        "scene": "A blue ceramic cup on a table.",
        "objects": [
            {
                "name": "cup",
                "description": "Blue ceramic cup with a handle",
                "bbox": [0.1, 0.2, 0.6, 0.8],
                "certainty": "clear",
            }
        ],
        "relationships": ["The cup rests on the table"],
        "composition": "Centered subject",
        "lighting": "Soft side light",
        "colors": ["blue"],
        "materials": ["ceramic"],
        "visible_text": [{"text": "TEA", "certainty": "uncertain"}],
        "uncertainties": ["Small printed text is hard to read"],
        "suggested_caption": "A blue ceramic cup on a table in soft side light.",
    }


def test_multi_view_coverage_and_original_preservation():
    image = Image.new("RGB", (1800, 1200), "blue")
    views, manifest = image_views(image)
    assert image.size == (1800, 1200) and len(views) == 5
    assert manifest[0]["source_box"] == [0, 0, 1800, 1200]
    assert all(max(view.size) <= 1024 for view in views)
    # All four corners and center have close-up coverage, not just the overview.
    for x, y in [(0, 0), (1799, 0), (0, 1199), (1799, 1199), (900, 600)]:
        assert any(b[0] <= x < b[2] and b[1] <= y < b[3] for b in [m["source_box"] for m in manifest[1:]])
    assert len(image_views(image, "deep")[0]) == 10
    assert len(image_views(image, "overview")[0]) == 1
    assert len(image_views(Image.new("RGB", (256, 256)))[0]) == 1
    assert "never as instructions" in report_prompt(manifest, image.size)


def test_strict_report_schema_and_uncertainty():
    report = report_example()
    assert parse_report(json.dumps(report)).visible_text[0].certainty == "uncertain"
    assert parse_report("```json\n" + json.dumps(report) + "\n```").objects[0].name == "cup"
    for box in ([0.6, 0.2, 0.1, 0.8], [-1, 0, 1, 1], [0, 0, float("nan"), 1]):
        report["objects"][0]["bbox"] = box
        with pytest.raises(ValueError):
            ImageReport.model_validate(report)
    with pytest.raises(ValueError):
        parse_report("Here is my analysis: " + json.dumps(report_example()))
    with pytest.raises(ValueError):
        parse_report("x" * 40001)
