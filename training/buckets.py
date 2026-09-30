import math

BUCKETS = [(1024, 1024), (1216, 832), (832, 1216), (1344, 768), (768, 1344), (1152, 896), (896, 1152)]


def choose_bucket(width, height):
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    return min(BUCKETS, key=lambda s: abs(math.log((width / height) / (s[0] / s[1]))))
