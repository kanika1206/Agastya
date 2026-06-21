from __future__ import annotations

import numpy as np

from agastya.eval.degrade import motion_blur, motion_blur_kernel


def test_motion_blur_kernel_is_normalized():
    kernel = motion_blur_kernel(15, 30.0)
    assert kernel.shape == (15, 15)
    assert np.isclose(kernel.sum(), 1.0)


def test_motion_blur_preserves_shape_and_dtype():
    image = (np.random.rand(48, 64, 3) * 255).astype(np.uint8)
    out = motion_blur(image, 15, 30.0)
    assert out.shape == image.shape
    assert out.dtype == image.dtype


def test_motion_blur_actually_blurs():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[16, 16] = 255
    out = motion_blur(image, 15, 0.0)
    assert int(out.max()) < 255


def test_motion_blur_is_deterministic():
    image = (np.random.rand(40, 40, 3) * 255).astype(np.uint8)
    a = motion_blur(image, 11, 45.0)
    b = motion_blur(image, 11, 45.0)
    assert np.array_equal(a, b)


def test_motion_blur_kernel_rejects_nonpositive_size():
    try:
        motion_blur_kernel(0, 0.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for kernel_size < 1")
