# cage_fit.py
from __future__ import annotations
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

try:
    from scipy.interpolate import splprep, splev
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


"""
Edge Fitting Module for Bearing Cage Images
===========================================

This module fits geometric models (ellipse or spline) to bearing cage edges
detected in high-speed camera images.

Workflow
--------
1. Preprocess edge image (morphological closing + dilation).
2. Extract contours from edges.
3. Rank contours by ellipse area (largest = outer, smaller = inner).
4. For each contour:
    - Attempt robust ellipse fitting (RANSAC).
    - Evaluate inlier ratio and RMSE.
    - If ellipse is valid → use ellipse model.
    - If ellipse fails → fit closed spline.
    - If spline fails → return raw polyline.
5. Label results ('outer', 'inner').
6. Optionally overlay fitted models for visualization.

Concept
-------
- Ellipse fitting provides a clean geometric approximation when possible.
- Spline fitting provides a flexible curve model if ellipse is unsuitable.
- This combination ensures robustness against imperfect or noisy edges.
"""


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class EllipseResult:

    """
        Result of ellipse fitting.

        Attributes
        ----------

        center : tuple
            (cx, cy) ellipse center.
        axes : tuple
            (w, h) full axis lengths in OpenCV convention.
        angle : float
            Ellipse orientation in degrees.
        inlier_ratio : float
            Fraction of contour points close to ellipse.
        rmse : float
            Root mean square error of inliers.
        type : str
        Fixed string 'ellipse'.
    """

    center: tuple  # (cx, cy)
    axes: tuple    # (w, h) full lengths as in OpenCV
    angle: float   # degrees
    inlier_ratio: float
    rmse: float
    type: str = "ellipse"

@dataclass
class SplineResult:

    """
    Result of closed spline fitting.

    Attributes
    ----------
    tck : object
        Spline representation from SciPy.

    samples : np.ndarray
        Sampled points along spline (N,2).

    smoothing : float
        Smoothing factor used.

    type : str
        Fixed string 'spline'.
    """

    tck: Any
    samples: np.ndarray  # (N,2)
    smoothing: float
    type: str = "spline"

# -----------------------------
# Geometry helpers
# -----------------------------

def _ellipse_F_residuals(points: np.ndarray, center, axes, angle_deg) -> np.ndarray:

    """
    Compute implicit ellipse residuals.

    Formula
    -------
    F = (x'/a)^2 + (y'/b)^2 - 1, where (x', y') are rotated coordinates.

    Parameters
    ----------
    points : np.ndarray, shape (N,2)
        Contour points.
    center : tuple
        Ellipse center (cx, cy).
    axes : tuple
        Ellipse axes (w, h) as full lengths.
    angle_deg : float
        Ellipse rotation in degrees.

    Returns
    -------
    np.ndarray
        Absolute residuals for each point.
    """
    cx, cy = center
    w, h = axes
    if w <= 0 or h <= 0:
        return np.full(len(points), np.inf)
    a = w / 2.0
    b = h / 2.0
    theta = np.deg2rad(angle_deg)
    # Rotate by -theta to align with ellipse axes
    c, s = np.cos(theta), np.sin(theta)
    Rm = np.array([[ c,  s],
                   [-s,  c]])
    P = points - np.array([cx, cy])
    Pp = P @ Rm.T
    val = (Pp[:, 0] / a) ** 2 + (Pp[:, 1] / b) ** 2 - 1.0
    return np.abs(val)

# -----------------------------
# Preprocess & contours
# -----------------------------

def preprocess_edges(edge_img: np.ndarray, pre_close_kernel: int = 3, pre_dilate_iter: int = 1) -> np.ndarray:

    """
    Preprocess edge image by morphological closing and dilation.

    Parameters
    ----------
    edge_img : np.ndarray
        Binary edge image (uint8 or convertible).
    pre_close_kernel : int, optional
        Size of structuring element for closing (default=3).
    pre_dilate_iter : int, optional
        Number of dilation iterations (default=1).

    Returns
    -------
    np.ndarray
        Processed binary image with closed gaps and thicker edges.
    """

    edge = edge_img.copy()
    if edge.dtype != np.uint8:
        edge = edge.astype(np.uint8)
    k = max(1, int(pre_close_kernel))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)
    if pre_dilate_iter > 0:
        closed = cv2.dilate(closed, kernel, iterations=int(pre_dilate_iter))
    return closed


def find_candidate_contours(edge_img: np.ndarray) -> List[np.ndarray]:

    """
    Find candidate contours from an edge image.

    Parameters
    ----------
    edge_img : np.ndarray
        Binary edge image.

    Returns
    -------
    List[np.ndarray]
        List of contours, each as float32 array (N,2).
    """

    cnts_info = cv2.findContours(edge_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = cnts_info[0] if len(cnts_info) == 2 else cnts_info[1]
    out = []
    for c in contours:
        c2 = c.reshape(-1, 2)
        if len(c2) >= 5:
            out.append(c2.astype(np.float32))
    return out


def rank_contours_by_ellipse_area(contours: List[np.ndarray]) -> List[np.ndarray]:
    """
    Rank contours by area of fitted ellipse.

    Parameters
    ----------
    contours : list of np.ndarray
        Candidate contours.

    Returns
    -------
    List[np.ndarray]
        Contours sorted by ellipse area (largest first).
    """
    scored = []
    for c in contours:
        area = 0.0
        try:
            ell = cv2.fitEllipse(c)
            (cx, cy), (w, h), ang = ell
            area = float(np.pi * (w / 2.0) * (h / 2.0))
        except cv2.error:
            # fallback to minEnclosingCircle area or contourArea
            (x0, y0), r = cv2.minEnclosingCircle(c.reshape(-1, 1, 2))
            area = float(np.pi * r * r)
        scored.append((area, c))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [c for _, c in scored]

# -----------------------------
# RANSAC ellipse fit (using cv2.fitEllipse as inner solver)
# -----------------------------

def ransac_ellipse_fit(points: np.ndarray,n_iters: int = 1000,sample_size: int = 30,residual_thresh: float = 0.03,
                       min_inliers_ratio: float = 0.5, rng: Optional[np.random.Generator] = None) -> Optional[EllipseResult]:
    """
    Fit ellipse using RANSAC for robustness.

    Parameters
    ----------
    points : np.ndarray, shape (N,2)
        Contour points.
    n_iters : int, optional
        Number of RANSAC iterations (default=1000).
    sample_size : int, optional
        Subset size for ellipse fitting (default=30).
    residual_thresh : float, optional
        Threshold for point-to-ellipse residual (default=0.03).
    min_inliers_ratio : float, optional
        Minimum inlier ratio to accept fit (default=0.5).
    rng : np.random.Generator, optional
        Random generator.

    Returns
    -------
    EllipseResult or None
        Fitted ellipse parameters or None if fit fails.
    """

    N = len(points)
    if N < 5:
        return None
    if rng is None:
        rng = np.random.default_rng()
    sample_size = max(5, min(sample_size, N))

    best_inliers_count = -1
    best_rmse = np.inf
    best_model = None
    best_inliers_mask = None

    for _ in range(int(n_iters)):
        idx = rng.choice(N, size=sample_size, replace=False)
        sample = points[idx]
        try:
            ell = cv2.fitEllipse(sample.astype(np.float32))
        except cv2.error:
            continue
        (cx, cy), (w, h), ang = ell
        if w <= 0 or h <= 0:
            continue
        res = _ellipse_F_residuals(points, (cx, cy), (w, h), ang)
        inliers_mask = res <= residual_thresh
        inliers_count = int(inliers_mask.sum())
        if inliers_count < 5:
            continue
        rmse = float(np.sqrt(np.mean(res[inliers_mask] ** 2)))
        if (inliers_count > best_inliers_count) or (
            inliers_count == best_inliers_count and rmse < best_rmse
        ):
            best_inliers_count = inliers_count
            best_rmse = rmse
            best_model = (cx, cy, w, h, ang)
            best_inliers_mask = inliers_mask

    if best_model is None:
        return None

    # Refine on inliers
    cx, cy, w, h, ang = best_model
    try:
        refined = cv2.fitEllipse(points[best_inliers_mask].astype(np.float32))
        (cx, cy), (w, h), ang = refined
    except cv2.error:
        pass

    res = _ellipse_F_residuals(points, (cx, cy), (w, h), ang)
    inliers_mask = res <= residual_thresh
    rmse = float(np.sqrt(np.mean(res[inliers_mask] ** 2)))
    inlier_ratio = float(inliers_mask.mean())

    if inlier_ratio < min_inliers_ratio:
        return None

    return EllipseResult(center=(float(cx), float(cy)),
                         axes=(float(w), float(h)),
                         angle=float(ang),
                         inlier_ratio=inlier_ratio,
                         rmse=rmse)

# -----------------------------
# Closed spline fit
# -----------------------------

def fit_closed_spline(contour_points: np.ndarray, smoothing: float = 2.0, num_samples: int = 400) -> SplineResult:

    """
    Fit a closed periodic spline to contour points.
    Parameters
    ----------
    contour_points : np.ndarray
        Contour points.
    smoothing : float, optional
        Smoothing factor (default=2.0).
    num_samples : int, optional
        Number of spline samples (default=400).

    Returns
    -------
    SplineResult
        Closed spline representation.
    """

    if not _HAS_SCIPY:
        raise RuntimeError("SciPy not available: install scipy for spline fitting.")

    pts = contour_points.astype(np.float64)
    # Remove consecutive duplicates
    if len(pts) > 1:
        diffs = np.diff(pts, axis=0)
        keep = np.any(diffs != 0, axis=1)
        pts = np.vstack([pts[0], pts[1:][keep]])

    # Ensure closed curve
    if np.linalg.norm(pts[0] - pts[-1]) > 1e-9:
        pts = np.vstack([pts, pts[0]])

    # Parameterize by cumulative arc length (0..1)
    dists = np.sqrt(((np.diff(pts, axis=0)) ** 2).sum(axis=1))
    u = np.hstack([[0.0], np.cumsum(dists)])
    total = u[-1]
    if total <= 0:
        total = 1.0
    u /= total

    # Closed periodic spline
    tck, _ = splprep([pts[:, 0], pts[:, 1]], u=u, s=float(smoothing), per=True)
    unew = np.linspace(0.0, 1.0, int(num_samples), endpoint=False)
    out = splev(unew, tck)
    samples = np.stack(out, axis=1)
    return SplineResult(tck=tck, samples=samples, smoothing=float(smoothing))

# -----------------------------
# Main orchestrator
# -----------------------------

def fit_cage_edges(
    edge_img: np.ndarray,
    max_regions: int = 2,
    pre_close_kernel: int = 3,
    pre_dilate_iter: int = 1,
    ransac_iters: int = 1000,
    sample_size: int = 30,
    residual_thresh: float = 0.03,
    min_inliers_ratio: float = 0.5,
    spline_smoothing: float = 2.0,
    num_spline_samples: int = 400,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict[str, Any]]:

    """
    Main function to fit outer and inner cage edges.
    Parameters
    ----------
    edge_img : np.ndarray
        Binary edge image.
    max_regions : int, optional
        Max number of contours to fit (default=2).
    pre_close_kernel : int, optional
        Kernel size for closing (default=3).
    pre_dilate_iter : int, optional
        Dilation iterations (default=1).
    ransac_iters : int, optional
        RANSAC iterations (default=1000).
    sample_size : int, optional
        Subset size for ellipse fitting (default=30).
    residual_thresh : float, optional
        Threshold for residuals (default=0.03).
    min_inliers_ratio : float, optional
        Minimum inlier ratio (default=0.5).
    spline_smoothing : float, optional
        Smoothing factor for spline (default=2.0).
    num_spline_samples : int, optional
        Number of spline samples (default=400).
    rng : np.random.Generator, optional
        Random generator.

    Returns
    -------
    List[Dict[str, Any]]
        List of fitted models with labels ('outer'/'inner').
    """

    pre = preprocess_edges(edge_img, pre_close_kernel, pre_dilate_iter)
    contours = find_candidate_contours(pre)
    if not contours:
        return []
    ranked = rank_contours_by_ellipse_area(contours)
    regions = []

    for i, c in enumerate(ranked[:max_regions]):
        label = 'outer' if i == 0 else 'inner'
        ell = ransac_ellipse_fit(c,
                                 n_iters=ransac_iters,
                                 sample_size=sample_size,
                                 residual_thresh=residual_thresh,
                                 min_inliers_ratio=min_inliers_ratio,
                                 rng=rng)
        if ell is not None and (ell.inlier_ratio >= min_inliers_ratio and ell.rmse <= residual_thresh * 1.5):
            regions.append({"label": label, "model": ell})
        else:
            try:
                sp = fit_closed_spline(c, smoothing=spline_smoothing, num_samples=num_spline_samples)
                regions.append({"label": label, "model": sp})
            except Exception:
                # Fallback: return polyline
                regions.append({"label": label, "model": {"type": "polyline", "points": c}})

    return regions

# -----------------------------
# Visualization helper
# -----------------------------

def overlay_fits(image: np.ndarray, results: List[Dict[str, Any]]) -> np.ndarray:
    """
    Overlay fitted models on an image for visualization.

    Parameters
    ----------
    image : np.ndarray
        Input image (grayscale or BGR).
    results : List[Dict[str, Any]]
        Results from fit_cage_edges.

    Returns
    -------
    np.ndarray
        Image with ellipses (red), splines (green), or polylines (yellow) drawn.
    """

    vis = image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    for r in results:
        m = r["model"]
        if isinstance(m, EllipseResult):
            ell = (m.center, m.axes, m.angle)
            cv2.ellipse(vis, ell, (0, 0, 255), 2)  # red
        elif isinstance(m, SplineResult):
            pts = m.samples.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)  # green
        elif isinstance(m, dict) and m.get("type") == "polyline":
            pts = m["points"].astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], isClosed=False, color=(255, 255, 0), thickness=1)
    return vis

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        raise SystemExit("Usage: python cage_fit.py <image_path>")

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Could not read image: {path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    results = fit_cage_edges(
        edges,
        max_regions=2,
        pre_close_kernel=3,
        pre_dilate_iter=1,
        ransac_iters=1000,
        sample_size=30,
        residual_thresh=0.03,
        min_inliers_ratio=0.5,
        spline_smoothing=2.0,
        num_spline_samples=400,
    )

    out = overlay_fits(img, results)
    cv2.imshow("Fits", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
