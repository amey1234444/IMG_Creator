from img_creator.config import PRESETS


def test_all_presets_have_positive_dimensions():
    for qualities in PRESETS.values():
        for preset in qualities.values():
            assert all(value > 0 for value in preset.base_size)
            assert all(value > 0 for value in preset.final_size)


def test_final_resolution_not_smaller_than_base():
    for qualities in PRESETS.values():
        for preset in qualities.values():
            assert preset.final_size[0] >= preset.base_size[0]
            assert preset.final_size[1] >= preset.base_size[1]
