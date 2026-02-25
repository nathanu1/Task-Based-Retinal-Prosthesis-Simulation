"""Improved saliency: Spectral Residual + OpenCV FineGrained + fusion."""

import numpy as np
import cv2
from typing import Optional, Literal

SaliencyMethod = Literal["spectral", "fine_grained", "combined"]


def _spectral_residual(gray: np.ndarray) -> np.ndarray:
    gray_f = np.float32(gray) / 255.0
    fft = np.fft.fft2(gray_f)
    log_amp = np.log(np.abs(fft) + 1e-8)
    phase = np.angle(fft)
    avg_log = cv2.blur(log_amp, (5, 5))
    residual = log_amp - avg_log
    sal_fft = np.exp(residual + 1j * phase)
    saliency = np.abs(np.fft.ifft2(sal_fft).real) ** 2
    saliency = cv2.GaussianBlur(saliency, (9, 9), 2)
    return saliency


def _fine_grained_opencv(image: np.ndarray) -> Optional[np.ndarray]:
    try:
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        success, sal_map = saliency.computeSaliency(image)
        if success and sal_map is not None:
            return np.clip(sal_map.astype(np.float32), 0, 1)
    except Exception:
        pass
    return None


def _objectness_bing(image: np.ndarray) -> Optional[np.ndarray]:
    try:
        saliency = cv2.saliency.ObjectnessBING_create()
        success, sal_map = saliency.computeSaliency(image)
        if success and sal_map is not None:
            return np.clip(sal_map.astype(np.float32), 0, 1)
    except Exception:
        pass
    return None


class SpectralResidualSaliency:
    """Spectral residual saliency detection."""

    def compute_saliency(self, frame: np.ndarray, method: SaliencyMethod = "combined") -> np.ndarray:
        """Compute saliency map [0, 1] from frame."""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        spectral = _spectral_residual(gray)
        spectral = (spectral - spectral.min()) / (spectral.max() - spectral.min() + 1e-8)

        if method == "spectral":
            return spectral.astype(np.float32)

        fine = _fine_grained_opencv(frame)
        if fine is not None and method == "combined":
            combined = 0.5 * spectral + 0.5 * fine
        elif fine is not None:
            return fine
        else:
            combined = spectral

        combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)
        combined = cv2.GaussianBlur(combined, (5, 5), 1)
        return np.clip(combined, 0, 1).astype(np.float32)
