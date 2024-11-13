# biotrack, Apache-2.0 license
# Filename: biotrack/tracker/assoc.py
# Description:  Track association functions

import numpy as np
from scipy.spatial.distance import cdist

from biotrack.logger import debug

def associate(detection_pts, detection_emb, tracker_pts, tracker_emb, alpha: float = 0.8) -> np.ndarray:
    if len(detection_pts) == 0 or len(tracker_pts) == 0:
        return [100.0 * np.ones(len(detection_pts), dtype=float)]

    debug("=======================")
    debug(f"detection_pts: {detection_pts}")
    debug("=======================")
    debug(f"tracker_pts: {tracker_pts}")
    debug("=======================")
    debug(f"detection_emb: {detection_emb.shape}")
    debug("=======================")
    debug(f"tracker_emb: {tracker_emb.shape}")
    # Calculate Euclidean distance and Cosine distance
    D_E = np.round(100 * cdist(detection_pts, tracker_pts, metric="euclidean"))
    D_C = cdist(detection_emb, tracker_emb, metric="cosine")
    np.nan_to_num(D_C, copy=False, nan=0.0)
    D_C = np.round(100 * D_C)
    D_combined = alpha * D_E + (1 - alpha) * D_C
    debug(f"D_E: {D_E}  - D_C: {D_C} D_combined: {D_combined}")
    return D_combined
