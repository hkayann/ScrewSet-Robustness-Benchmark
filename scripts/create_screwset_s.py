#!/usr/bin/env python3
"""
Generate ScrewSet-S (Simulated Corruptions) dataset.

This script applies the same 19 corruption types from CIFAR-10-C / ImageNet-C
to the ScrewSet test set, creating a simulated corruption benchmark for
direct comparison with the physical corruptions in ScrewSet-C.

Output format matches CIFAR-10-C:
- Each corruption → {corruption_name}.npy with shape (N*5, H, W, 3)
- labels.npy with shape (N*5,) containing class indices repeated 5x

Where N = number of test images, 5 = severity levels.

Reference: Hendrycks & Dietterich, "Benchmarking Neural Network Robustness
to Common Corruptions and Perturbations", ICLR 2019.
"""

import os
import sys
import numpy as np
from PIL import Image as PILImage
from pathlib import Path
from tqdm import tqdm
import cv2
from io import BytesIO
from scipy.ndimage import zoom as scizoom
from scipy.ndimage import map_coordinates
from skimage.filters import gaussian
import skimage as sk
import ctypes
import warnings
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from numba import jit, prange

warnings.simplefilter("ignore", UserWarning)

try:
    from wand.image import Image as WandImage
    from wand.api import library as wandlibrary
    HAVE_WAND = True
except Exception:
    HAVE_WAND = False

# ============================================================================
# Configuration
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
SCREWSET_TEST = DATA_DIR / "screwset_split" / "test"
OUTPUT_DIR = DATA_DIR / "screwset_s"

# ScrewSet image dimensions
IMG_HEIGHT = 480
IMG_WIDTH = 640

# All 19 corruption types from CIFAR-10-C / ImageNet-C
CORRUPTION_NAMES = [
    # Noise
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "speckle_noise",
    # Blur
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "gaussian_blur",
    # Weather
    "snow",
    "frost",
    "fog",
    "brightness",
    # Digital
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
    "spatter",
    "saturate",
]

# ============================================================================
# Corruption Helpers (adapted from hendrycks/robustness)
# ============================================================================

def disk(radius, alias_blur=0.1, dtype=np.float32):
    """Create a disk-shaped kernel for defocus blur."""
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


# Setup wand motion blur
if HAVE_WAND:
    wandlibrary.MagickMotionBlurImage.argtypes = (
        ctypes.c_void_p,  # wand
        ctypes.c_double,  # radius
        ctypes.c_double,  # sigma
        ctypes.c_double,  # angle
    )


class MotionImage(WandImage if HAVE_WAND else object):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        if HAVE_WAND:
            wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


def plasma_fractal(mapsize=512, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Returns square 2d array of floats in range 0-1.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float64)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def clipped_zoom(img, zoom_factor):
    """Zoom into an image while maintaining the original size - optimized with cv2."""
    h, w = img.shape[:2]
    ch = int(np.ceil(h / float(zoom_factor)))
    cw = int(np.ceil(w / float(zoom_factor)))

    top_h = (h - ch) // 2
    top_w = (w - cw) // 2
    
    cropped = img[top_h:top_h + ch, top_w:top_w + cw]
    
    # Use cv2.resize instead of scipy.ndimage.zoom (much faster)
    if len(img.shape) == 3:
        zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return zoomed


# ============================================================================
# Corruption Functions (ImageNet-C severity parameters, adapted for 640x480)
# ============================================================================

def gaussian_noise(x, severity=1):
    """Add Gaussian noise."""
    c = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
    x = np.array(x) / 255.0
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def shot_noise(x, severity=1):
    """Add shot (Poisson) noise."""
    c = [60, 25, 12, 5, 3][severity - 1]
    x = np.array(x) / 255.0
    return np.clip(np.random.poisson(x * c) / float(c), 0, 1) * 255


def impulse_noise(x, severity=1):
    """Add salt-and-pepper noise."""
    c = [0.03, 0.06, 0.09, 0.17, 0.27][severity - 1]
    x = sk.util.random_noise(np.array(x) / 255.0, mode='s&p', amount=c)
    return np.clip(x, 0, 1) * 255


def speckle_noise(x, severity=1):
    """Add multiplicative speckle noise."""
    c = [0.15, 0.2, 0.35, 0.45, 0.6][severity - 1]
    x = np.array(x) / 255.0
    return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def gaussian_blur(x, severity=1):
    """Apply Gaussian blur."""
    c = [1, 2, 3, 4, 6][severity - 1]
    x = gaussian(np.array(x) / 255.0, sigma=c, channel_axis=-1)
    return np.clip(x, 0, 1) * 255


@jit(nopython=True, parallel=True)
def _glass_blur_shuffle(x, max_delta, iterations):
    """Fast pixel shuffling using numba JIT compilation."""
    h, w, c = x.shape
    result = x.copy()
    
    for _ in range(iterations):
        # Generate all random offsets at once
        dx = np.random.randint(-max_delta, max_delta + 1, size=(h, w))
        dy = np.random.randint(-max_delta, max_delta + 1, size=(h, w))
        
        # Create output for this iteration
        new_result = result.copy()
        
        for row in prange(max_delta, h - max_delta):
            for col in range(max_delta, w - max_delta):
                h_prime = row + dy[row, col]
                w_prime = col + dx[row, col]
                # Bounds check
                if 0 <= h_prime < h and 0 <= w_prime < w:
                    for ch in range(c):
                        new_result[row, col, ch] = result[h_prime, w_prime, ch]
        
        result = new_result
    
    return result


def glass_blur(x, severity=1):
    """Apply glass blur (Gaussian + local pixel shuffling) - optimized with numba."""
    # sigma, max_delta, iterations - scaled for larger images
    c = [(0.7, 2, 2), (0.9, 3, 1), (1.0, 3, 3), (1.1, 4, 2), (1.5, 5, 2)][severity - 1]
    
    x = np.uint8(gaussian(np.array(x) / 255.0, sigma=c[0], channel_axis=-1) * 255)
    
    # Use numba-accelerated shuffling
    x = _glass_blur_shuffle(x, c[1], c[2])
    
    return np.clip(gaussian(x / 255.0, sigma=c[0], channel_axis=-1), 0, 1) * 255


def defocus_blur(x, severity=1):
    """Apply defocus (disk) blur."""
    c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]
    
    x = np.array(x) / 255.0
    kernel = disk(radius=c[0], alias_blur=c[1])
    
    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))
    
    return np.clip(channels, 0, 1) * 255


def motion_blur(x, severity=1):
    """Apply motion blur using ImageMagick."""
    # Scaled for larger images
    c = [(12, 4), (18, 6), (18, 10), (18, 14), (24, 18)][severity - 1]

    if not HAVE_WAND:
        ksize = max(3, int(c[0]))
        if ksize % 2 == 0:
            ksize += 1
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        kernel[ksize // 2, :] = 1.0
        angle = np.random.uniform(-45, 45)
        rot = cv2.getRotationMatrix2D((ksize // 2, ksize // 2), angle, 1.0)
        kernel = cv2.warpAffine(kernel, rot, (ksize, ksize))
        kernel_sum = np.sum(kernel)
        if kernel_sum > 0:
            kernel /= kernel_sum
        result = cv2.filter2D(np.uint8(x), -1, kernel)
        return np.clip(result, 0, 255)
    
    output = BytesIO()
    PILImage.fromarray(np.uint8(x)).save(output, format='PNG')
    img = MotionImage(blob=output.getvalue())
    
    img.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))
    
    result = cv2.imdecode(np.frombuffer(img.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED)
    
    if result.shape[2] == 4:  # BGRA
        result = cv2.cvtColor(result, cv2.COLOR_BGRA2RGB)
    elif len(result.shape) == 3:  # BGR
        result = result[..., [2, 1, 0]]  # BGR to RGB
    
    return np.clip(result, 0, 255)


def zoom_blur(x, severity=1):
    """Apply zoom blur."""
    c = [
        np.arange(1, 1.11, 0.01),
        np.arange(1, 1.16, 0.01),
        np.arange(1, 1.21, 0.02),
        np.arange(1, 1.26, 0.02),
        np.arange(1, 1.31, 0.03),
    ][severity - 1]
    
    x = (np.array(x) / 255.0).astype(np.float32)
    out = np.zeros_like(x)
    
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)
    
    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255


def fog(x, severity=1):
    """Add fog effect."""
    c = [(1.5, 2), (2.0, 2), (2.5, 1.7), (2.5, 1.5), (3.0, 1.4)][severity - 1]
    
    h, w = x.shape[:2]
    # Use power of 2 mapsize >= max(h, w)
    mapsize = 1
    while mapsize < max(h, w):
        mapsize *= 2
    
    x = np.array(x) / 255.0
    max_val = x.max()
    
    fog_layer = plasma_fractal(mapsize=mapsize, wibbledecay=c[1])[:h, :w]
    x += c[0] * fog_layer[..., np.newaxis]
    
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


def frost(x, severity=1):
    """Add frost effect using generated patterns."""
    c = [(1, 0.4), (0.8, 0.6), (0.7, 0.7), (0.65, 0.7), (0.6, 0.75)][severity - 1]
    
    h, w = x.shape[:2]
    
    # Generate frost-like pattern instead of using external images
    # Create a crystalline frost pattern using noise and blurring
    np.random.seed(np.random.randint(0, 10000))
    frost_layer = np.random.normal(0.5, 0.3, (h, w, 3))
    frost_layer = gaussian(frost_layer, sigma=3, channel_axis=-1)
    frost_layer = np.clip(frost_layer * 255, 0, 255).astype(np.uint8)
    
    # Add some structure
    frost_layer = cv2.blur(frost_layer, (15, 15))
    
    return np.clip(c[0] * np.array(x) + c[1] * frost_layer, 0, 255)


def snow(x, severity=1):
    """Add snow effect."""
    c = [
        (0.1, 0.3, 3, 0.5, 10, 4, 0.8),
        (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
        (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
        (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
        (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55),
    ][severity - 1]
    
    h, w = x.shape[:2]
    x = np.array(x, dtype=np.float32) / 255.0
    
    snow_layer = np.random.normal(size=(h, w), loc=c[0], scale=c[1])
    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2]).squeeze()
    snow_layer[snow_layer < c[3]] = 0
    
    # Motion blur the snow
    snow_uint8 = (np.clip(snow_layer, 0, 1) * 255).astype(np.uint8)
    if HAVE_WAND:
        snow_pil = PILImage.fromarray(snow_uint8, mode='L')
        output = BytesIO()
        snow_pil.save(output, format='PNG')
        snow_img = MotionImage(blob=output.getvalue())
        snow_img.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))
        snow_layer = cv2.imdecode(np.frombuffer(snow_img.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED) / 255.0
    else:
        # OpenCV fallback for motion blur on snow layer
        ksize = max(3, int(c[4]))
        if ksize % 2 == 0:
            ksize += 1
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        kernel[ksize // 2, :] = 1.0
        angle = np.random.uniform(-135, -45)
        rot = cv2.getRotationMatrix2D((ksize // 2, ksize // 2), angle, 1.0)
        kernel = cv2.warpAffine(kernel, rot, (ksize, ksize))
        kernel_sum = np.sum(kernel)
        if kernel_sum > 0:
            kernel /= kernel_sum
        snow_uint8 = cv2.filter2D(snow_uint8, -1, kernel)
        snow_layer = snow_uint8.astype(np.float64) / 255.0
    
    # Ensure correct shape
    if snow_layer.shape[:2] != (h, w):
        snow_layer = cv2.resize(snow_layer, (w, h))
    snow_layer = snow_layer[..., np.newaxis]
    
    gray = cv2.cvtColor((x * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).reshape(h, w, 1) / 255.0
    x = c[6] * x + (1 - c[6]) * np.maximum(x, gray * 1.5 + 0.5)
    
    return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255


def brightness(x, severity=1):
    """Increase brightness."""
    c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    
    x = np.array(x) / 255.0
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)
    
    return np.clip(x, 0, 1) * 255


def contrast(x, severity=1):
    """Reduce contrast."""
    c = [0.4, 0.3, 0.2, 0.1, 0.05][severity - 1]
    
    x = np.array(x) / 255.0
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def saturate(x, severity=1):
    """Adjust saturation."""
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]
    
    x = np.array(x) / 255.0
    x = sk.color.rgb2hsv(x)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    x = sk.color.hsv2rgb(x)
    
    return np.clip(x, 0, 1) * 255


def jpeg_compression(x, severity=1):
    """Apply JPEG compression artifacts."""
    c = [25, 18, 15, 10, 7][severity - 1]
    
    output = BytesIO()
    PILImage.fromarray(np.uint8(x)).save(output, 'JPEG', quality=c)
    output.seek(0)
    return np.array(PILImage.open(output))


def pixelate(x, severity=1):
    """Apply pixelation."""
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
    
    h, w = x.shape[:2]
    img = PILImage.fromarray(np.uint8(x))
    img = img.resize((int(w * c), int(h * c)), PILImage.BOX)
    img = img.resize((w, h), PILImage.BOX)
    
    return np.array(img)


def elastic_transform(image, severity=1):
    """Apply elastic deformation - optimized with cv2.remap."""
    # Parameters scaled for 640x480 (roughly 2.8x ImageNet's 224)
    scale = 640 / 224  # ~2.86
    c = [
        (244 * 2 * scale, 244 * 0.7, 244 * 0.1 * scale),
        (244 * 2 * scale, 244 * 0.08, 244 * 0.2 * scale),
        (244 * 0.05 * scale, 244 * 0.01, 244 * 0.02 * scale),
        (244 * 0.07 * scale, 244 * 0.01, 244 * 0.02 * scale),
        (244 * 0.12 * scale, 244 * 0.01, 244 * 0.02 * scale),
    ][severity - 1]
    
    image = np.array(image, dtype=np.float32) / 255.0
    shape = image.shape
    shape_size = shape[:2]
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([
        center_square + square_size,
        [center_square[0] + square_size, center_square[1] - square_size],
        center_square - square_size,
    ])
    pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    
    dx = (gaussian(np.random.uniform(-1, 1, size=shape[:2]), c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dy = (gaussian(np.random.uniform(-1, 1, size=shape[:2]), c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    
    # Use cv2.remap instead of scipy.ndimage.map_coordinates (much faster)
    h, w = shape[:2]
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x_coords + dx).astype(np.float32)
    map_y = (y_coords + dy).astype(np.float32)
    
    result = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    
    return np.clip(result, 0, 1) * 255


def spatter(x, severity=1):
    """Add spatter (water droplets or mud)."""
    c = [
        (0.65, 0.3, 4, 0.69, 0.6, 0),
        (0.65, 0.3, 3, 0.68, 0.6, 0),
        (0.65, 0.3, 2, 0.68, 0.5, 0),
        (0.65, 0.3, 1, 0.65, 1.5, 1),
        (0.67, 0.4, 1, 0.65, 1.5, 1),
    ][severity - 1]
    
    x = np.array(x, dtype=np.float32) / 255.0
    h, w = x.shape[:2]
    
    liquid_layer = np.random.normal(size=(h, w), loc=c[0], scale=c[1])
    liquid_layer = gaussian(liquid_layer, sigma=c[2])
    liquid_layer[liquid_layer < c[3]] = 0
    
    if c[5] == 0:  # Water spatter
        liquid_layer = (liquid_layer * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_layer, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, ker)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)
        
        m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
        m /= np.max(m, axis=(0, 1)) + 1e-8
        m *= c[4]
        
        # Water is pale turquoise
        color = np.concatenate((
            175 / 255.0 * np.ones_like(m[..., :1]),
            238 / 255.0 * np.ones_like(m[..., :1]),
            238 / 255.0 * np.ones_like(m[..., :1]),
        ), axis=2)
        
        color = cv2.cvtColor(color.astype(np.float32), cv2.COLOR_BGR2BGRA)
        x_bgra = cv2.cvtColor(x.astype(np.float32), cv2.COLOR_BGR2BGRA)
        
        return cv2.cvtColor(np.clip(x_bgra + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255
    else:  # Mud spatter
        m = np.where(liquid_layer > c[3], 1, 0)
        m = gaussian(m.astype(np.float32), sigma=c[4])
        m[m < 0.8] = 0
        
        # Mud brown
        color = np.concatenate((
            63 / 255.0 * np.ones_like(x[..., :1]),
            42 / 255.0 * np.ones_like(x[..., :1]),
            20 / 255.0 * np.ones_like(x[..., :1]),
        ), axis=2)
        
        color *= m[..., np.newaxis]
        x *= (1 - m[..., np.newaxis])
        
        return np.clip(x + color, 0, 1) * 255


# Mapping from corruption names to functions
CORRUPTION_FUNCTIONS = {
    "gaussian_noise": gaussian_noise,
    "shot_noise": shot_noise,
    "impulse_noise": impulse_noise,
    "speckle_noise": speckle_noise,
    "defocus_blur": defocus_blur,
    "glass_blur": glass_blur,
    "motion_blur": motion_blur,
    "zoom_blur": zoom_blur,
    "gaussian_blur": gaussian_blur,
    "snow": snow,
    "frost": frost,
    "fog": fog,
    "brightness": brightness,
    "contrast": contrast,
    "elastic_transform": elastic_transform,
    "pixelate": pixelate,
    "jpeg_compression": jpeg_compression,
    "spatter": spatter,
    "saturate": saturate,
}


# ============================================================================
# Dataset Loading
# ============================================================================

def load_screwset_test():
    """
    Load ScrewSet test set images and labels.
    Returns:
        images: list of numpy arrays (H, W, 3) uint8
        labels: list of int class indices
        class_names: list of class name strings
    """
    test_dir = SCREWSET_TEST
    class_names = sorted([d for d in os.listdir(test_dir) if os.path.isdir(test_dir / d)])
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    images = []
    labels = []
    
    print(f"Loading ScrewSet test set from {test_dir}")
    print(f"Found {len(class_names)} classes")
    
    for class_name in tqdm(class_names, desc="Loading classes"):
        class_dir = test_dir / class_name
        for img_name in sorted(os.listdir(class_dir)):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')):
                img_path = class_dir / img_name
                img = PILImage.open(img_path).convert('RGB')
                img_array = np.array(img)
                images.append(img_array)
                labels.append(class_to_idx[class_name])
    
    print(f"Loaded {len(images)} images")
    return images, labels, class_names


# ============================================================================
# Corruption Application
# ============================================================================

def apply_corruption_to_image(args):
    """Apply a single corruption to a single image (for multiprocessing)."""
    img, corruption_name, severity, seed = args
    np.random.seed(seed)
    corruption_fn = CORRUPTION_FUNCTIONS[corruption_name]
    try:
        corrupted = corruption_fn(img, severity)
        return np.uint8(np.clip(corrupted, 0, 255))
    except Exception as e:
        print(f"Error applying {corruption_name} severity {severity}: {e}")
        return img  # Return original on error


def generate_corruption(images, labels, corruption_name, output_dir, num_workers=4):
    """
    Generate a single corruption type for all images at all severity levels.
    Saves to {output_dir}/{corruption_name}.npy
    """
    output_path = output_dir / f"{corruption_name}.npy"
    
    # Skip if already exists
    if output_path.exists():
        print(f"\nSkipping {corruption_name} (already exists at {output_path})")
        existing = np.load(output_path, mmap_mode='r')
        return existing.shape
    
    n_images = len(images)
    h, w = images[0].shape[:2]
    
    # CIFAR-10-C layout: concatenate severity 1..5, each containing N images
    output_path = output_dir / f"{corruption_name}.npy"
    corrupted_all = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.uint8,
        shape=(n_images * 5, h, w, 3),
    )

    print(f"\nGenerating {corruption_name} (severities 1..5)...")

    for severity in range(1, 6):
        start = (severity - 1) * n_images
        end = severity * n_images
        args_list = [
            (images[i], corruption_name, severity, np.random.randint(0, 2**31))
            for i in range(n_images)
        ]

        for i, args in enumerate(tqdm(args_list, desc=f"  Severity {severity}", leave=False)):
            corrupted_all[start + i] = apply_corruption_to_image(args)
    
    # Flush memmap to disk
    corrupted_all.flush()
    del corrupted_all
    print(f"  Saved to {output_path}")
    print(f"  Shape: {(n_images * 5, h, w, 3)}, dtype: uint8")

    return (n_images * 5, h, w, 3)


def main():
    parser = argparse.ArgumentParser(description="Generate ScrewSet-S (Simulated Corruptions)")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory for .npy files")
    parser.add_argument("--corruptions", type=str, default=None,
                        help="Comma-separated list of corruption names (default: all)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of parallel workers")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which corruptions to generate
    if args.corruptions:
        corruption_names = [c.strip() for c in args.corruptions.split(",")]
        for name in corruption_names:
            if name not in CORRUPTION_FUNCTIONS:
                print(f"Unknown corruption: {name}")
                print(f"Available: {list(CORRUPTION_FUNCTIONS.keys())}")
                sys.exit(1)
    else:
        corruption_names = CORRUPTION_NAMES
    
    # Load dataset
    images, labels, class_names = load_screwset_test()
    
    # Save labels in CIFAR-10-C layout (repeat labels for severity 1..5)
    labels_array = np.tile(np.array(labels, dtype=np.uint8), 5)
    labels_path = output_dir / "labels.npy"
    np.save(labels_path, labels_array)
    print(f"Saved labels to {labels_path}, shape: {labels_array.shape}")
    
    # Save class names mapping
    class_mapping = {i: name for i, name in enumerate(class_names)}
    with open(output_dir / "class_mapping.json", "w") as f:
        json.dump(class_mapping, f, indent=2)
    print(f"Saved class mapping to {output_dir / 'class_mapping.json'}")
    
    # Generate each corruption
    results = {}
    for corruption_name in corruption_names:
        shape = generate_corruption(images, labels, corruption_name, output_dir, args.num_workers)
        results[corruption_name] = {"shape": shape}
    
    # Save summary
    summary = {
        "dataset": "ScrewSet-S",
        "description": "Simulated corruptions applied to ScrewSet test set (same format as CIFAR-10-C / ImageNet-C), severities 1-5",
        "num_images": len(images),
        "num_classes": len(class_names),
        "image_shape": [IMG_HEIGHT, IMG_WIDTH, 3],
        "severity_levels": [1, 2, 3, 4, 5],
        "corruptions": results,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"ScrewSet-S generation complete!")
    print(f"Output directory: {output_dir}")
    print(f"Total corruptions: {len(corruption_names)}")
    print(f"Total .npy files: {len(corruption_names) + 1} (corruptions + labels)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
