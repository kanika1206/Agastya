from __future__ import annotations

import cv2
import numpy as np


def motion_blur_kernel(kernel_size: int, angle_deg: float) -> np.ndarray:
    if kernel_size < 1:
        raise ValueError("kernel_size must be >= 1")
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = (kernel_size - 1) / 2.0
    kernel[int(round(center)), :] = 1.0
    rot = cv2.getRotationMatrix2D((center, center), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (kernel_size, kernel_size))
    total = kernel.sum()
    if total <= 0:
        raise ValueError("degenerate motion kernel")
    return kernel / total


def motion_blur(image: np.ndarray, kernel_size: int = 15, angle_deg: float = 30.0) -> np.ndarray:
    kernel = motion_blur_kernel(kernel_size, angle_deg)
    return cv2.filter2D(image, -1, kernel)
