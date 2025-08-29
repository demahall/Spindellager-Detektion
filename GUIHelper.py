# cage_core.py

import numpy as np
import cv2

from FittingEdge import fit_cage_edges, overlay_fits  # from earlier

# ---------- basic helpers ----------
def fit_circle_from_points(points):
    pts = np.asarray(points, dtype=float)
    x = pts[:,0]; y = pts[:,1]
    A = np.c_[2*x, 2*y, np.ones(len(pts))]
    b = x**2 + y**2
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    xc, yc = c[0], c[1]
    r = np.sqrt(c[2] + xc**2 + yc**2)
    return float(xc), float(yc), float(r)

def ring_xy(center, r, n=400):
    xc, yc = center
    ang = np.linspace(0, 2*np.pi, n)
    return xc + r*np.cos(ang), yc + r*np.sin(ang)

def build_annulus_mask(center, ref_point, delta_diam_px, shape_hw):
    """Returns mask(bool HxW), r_inner, r_outer."""
    (h, w) = shape_hw
    (xc, yc) = center
    (xr, yr) = ref_point
    r_ref = float(np.hypot(xr - xc, yr - yc))
    half = max(1.0, float(delta_diam_px) / 2.0)
    r_in = max(1.0, r_ref - half)
    r_out = r_ref + half
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((xx - xc)**2 + (yy - yc)**2)
    mask = (rr >= r_in) & (rr <= r_out)
    return mask, r_in, r_out

# ---------- processing ----------
def robust_normalize(gray2d, cutoff_low, cutoff_high, robust_fn):
    """gray2d -> normalized by your robust_contrast_normalization function."""
    norm_img, stretch = robust_fn(gray2d, cutoff_percentage=(float(cutoff_low), float(cutoff_high)))
    return norm_img, stretch

def canny_roi(src2d, roi_mask, low_thr, high_thr):
    if src2d.dtype != np.uint8:
        src2d = cv2.normalize(src2d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edges = cv2.Canny(src2d, int(low_thr), int(high_thr))
    edges[~roi_mask] = 0
    return edges

def fit_and_overlay(base_gray, edges, **fit_kwargs):
    results = fit_cage_edges(edges, **fit_kwargs)
    vis_bgr = overlay_fits(cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR), results)
    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    return results, vis_rgb

