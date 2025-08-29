
import json, time



# ---------- quality ----------
def _rasterize_model_mask(model, shape_hw):
    H, W = shape_hw
    mask = np.zeros((H, W), dtype=np.uint8)
    t = getattr(model, "type", "")
    if t == "ellipse":
        ell = (model.center, model.axes, model.angle)
        cv2.ellipse(mask, ell, 1, 1)
    elif t == "spline":
        pts = model.samples.astype(np.int32).reshape(-1,1,2)
        cv2.polylines(mask, [pts], True, 1, 1)
    elif isinstance(model, dict) and model.get("type") == "polyline":
        pts = model["points"].astype(np.int32).reshape(-1,1,2)
        cv2.polylines(mask, [pts], False, 1, 1)
    return mask

def evaluate_fit_quality(results, edges, tol_px=3.0):
    H, W = edges.shape
    edge_bin = (edges > 0).astype(np.uint8)
    if edge_bin.sum() == 0:
        return {}
    dt_edges = cv2.distanceTransform(1 - edge_bin, cv2.DIST_L2, 3)
    out = {}
    for r in results:
        lbl = r.get("label","?")
        m = r["model"]
        model_mask = _rasterize_model_mask(m, (H, W))
        if model_mask.sum() == 0:
            out[lbl] = {}
            continue
        d_me = dt_edges[model_mask > 0]
        mean_md = float(np.mean(d_me)); p95_md = float(np.percentile(d_me, 95))
        inlier_md = float(np.mean(d_me <= tol_px))

        dt_model = cv2.distanceTransform(1 - model_mask, cv2.DIST_L2, 3)
        d_em = dt_model[edge_bin > 0]
        mean_em = float(np.mean(d_em)); p95_em = float(np.percentile(d_em, 95))
        inlier_em = float(np.mean(d_em <= tol_px))

        out[lbl] = {
            "mean_md": mean_md, "p95_md": p95_md, "inlier_md": inlier_md,
            "mean_em": mean_em, "p95_em": p95_em, "inlier_em": inlier_em,
            "tol_px": float(tol_px)
        }
    return out



import numpy as np, cv2
from math import pi

# already have: _rasterize_model_mask, evaluate_fit_quality(...)

def ellipse_area_from_axes(axes):
    w, h = float(axes[0]), float(axes[1])
    return pi * (w/2.0) * (h/2.0)

def polygon_area(points):
    # points: (N,2) in order, closed or open
    P = np.asarray(points, dtype=float)
    if len(P) < 3: return 0.0
    if np.linalg.norm(P[0]-P[-1]) > 1e-9:
        P = np.vstack([P, P[0]])
    x, y = P[:,0], P[:,1]
    return 0.5 * float(np.sum(x[:-1]*y[1:] - x[1:]*y[:-1]))

def model_area_and_equiv_diam(model):
    t = getattr(model, "type", "")
    if t == "ellipse":
        A = ellipse_area_from_axes(model.axes)
    elif t == "spline":
        A = abs(polygon_area(model.samples))
    elif isinstance(model, dict) and model.get("type") == "polyline":
        A = abs(polygon_area(model["points"]))
    else:
        A = 0.0
    D_eq = 2.0 * np.sqrt(abs(A) / pi) if A > 0 else 0.0
    return float(abs(A)), float(D_eq)

def _containment_fraction(model_inner, model_outer, shape_hw):
    H, W = shape_hw
    mask_in  = _rasterize_model_mask(model_inner, (H, W))
    mask_out = _rasterize_model_mask(model_outer, (H, W))
    # Thicken to 2px band to be tolerant; then fill to region by morphological close
    band_in  = cv2.dilate(mask_in, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1)
    band_out = cv2.dilate(mask_out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1)
    # Fill: approximate interior using distance transform (fast)
    # Alternatively, cv2.findContours and fillPoly could be used.
    region_in  = cv2.distanceTransform(1-band_in, cv2.DIST_L2, 3) == 0  # band marks boundary; interior≈zeros
    region_out = cv2.distanceTransform(1-band_out, cv2.DIST_L2, 3) == 0
    inter = np.logical_and(region_in, region_out).sum()
    tot   = region_in.sum()
    return 0.0 if tot == 0 else inter / tot

def classify_inner_outer(results, shape_hw, min_containment=0.95):
    """
    Returns:
      classified (list like results, 'label' fields set to 'inner'/'outer'),
      stats { 'outer': {'area','D_eq'}, 'inner': {...}, 'containment': f, 'ok': bool, 'reason': str }
    """
    if not results or len(results) < 2:
        return results, {'ok': False, 'reason': 'Need two boundaries'}
    # compute areas
    items = []
    for r in results:
        A, Deq = model_area_and_equiv_diam(r['model'])
        items.append({'A': A, 'D': Deq, 'r': r})
    items.sort(key=lambda k: k['A'], reverse=True)
    outer, inner = items[0]['r'], items[1]['r']
    # containment
    frac = _containment_fraction(inner['model'], outer['model'], shape_hw)
    ok = frac >= float(min_containment)
    # assign labels
    outlist = []
    for r in results:
        lab = 'outer' if r is outer else ('inner' if r is inner else 'other')
        outlist.append({'label': lab, 'model': r['model']})
    stats = {
        'outer': {'area': model_area_and_equiv_diam(outer['model'])[0],
                  'D_eq': model_area_and_equiv_diam(outer['model'])[1]},
        'inner': {'area': model_area_and_equiv_diam(inner['model'])[0],
                  'D_eq': model_area_and_equiv_diam(inner['model'])[1]},
        'containment': float(frac),
        'ok': bool(ok),
        'reason': '' if ok else f'Inner not contained enough in outer (frac={frac:.2f})'
    }
    return outlist, stats

def detection_success(quality_dict, stats, tol_px=3.0,
                      min_inlier=0.60, max_p95=5.0):
    """
    Decide success from quality + containment.
    quality_dict: from evaluate_fit_quality
    """
    if not stats.get('ok', False):
        return False, stats.get('reason','Containment check failed')

    labels = ['outer','inner']
    for lbl in labels:
        q = quality_dict.get(lbl) or quality_dict.get(lbl.capitalize()) or quality_dict.get(lbl.title())
        if not q:
            return False, f'Missing quality for {lbl}'
        # inlier both ways at tol
        if q.get('inlier_md',0.0) < min_inlier or q.get('inlier_em',0.0) < min_inlier:
            return False, f'{lbl} inlier < {min_inlier*100:.0f}% @ {tol_px}px'
        # optional: p95 caps
        if q.get('p95_md', 1e9) > max_p95 or q.get('p95_em', 1e9) > max_p95:
            return False, f'{lbl} p95 too large (> {max_p95}px)'
    return True, 'Inner and outer cages detected with sufficient quality'


# ---------- save ----------
def build_session_dict(state):
    """state is a dict the GUI passes in with all fields it wants to persist."""
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        **state
    }

def save_session_json(path, session_dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session_dict, f, indent=2)
