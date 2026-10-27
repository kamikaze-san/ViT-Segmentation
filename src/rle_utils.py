import numpy as np


def rle_decode(rle_string, shape=(256, 1600)):
    """Decode a Kaggle-style RLE string into a binary mask.

    Kaggle RLE convention (verified against Severstal train.csv):
    - pairs of (start, length), 1-indexed pixel positions
    - pixel index increases down columns first (Fortran / column-major order)
    """
    if not isinstance(rle_string, str) or rle_string.strip() == "":
        return np.zeros(shape, dtype=np.uint8)

    s = list(map(int, rle_string.split()))
    starts, lengths = s[0::2], s[1::2]
    starts = np.array(starts) - 1  # to 0-indexed
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for st, ln in zip(starts, lengths):
        mask[st:st + ln] = 1
    return mask.reshape(shape, order="F")


def rle_encode(mask):
    """Encode a binary mask (H, W) into a Kaggle-style RLE string."""
    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


if __name__ == "__main__":
    # sanity check: round-trip a known example
    test_rle = "1 3 10 5"
    mask = rle_decode(test_rle, shape=(4, 4))
    assert mask[0, 0] == 1 and mask[1, 0] == 1 and mask[2, 0] == 1  # pixels 1,2,3 -> col-major (0,0),(1,0),(2,0)
    recovered = rle_encode(mask)
    assert recovered == test_rle, f"round-trip failed: {recovered} != {test_rle}"
    print("rle_decode/rle_encode round-trip OK")
