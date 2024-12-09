# biotrack, CC-BY-NC license
# Filename: biotrack/tracker/assoc.py
# Description:  Track association functions

import numpy as np
from scipy.spatial.distance import cdist
from biotrack.logger import debug, info, err
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity

from biotrack.logger import debug

def associate(detection_pts, detection_emb, tracker_pts, tracker_emb, alpha: float = 0.5) -> np.ndarray:
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
    if np.isnan(D_C).any():
        D_combined = D_E
    else:
        D_C = np.round(100 * D_C)
        D_combined = alpha * D_E + (1 - alpha) * D_C
    debug(f"D_E: {D_E}  - D_C: {D_C} D_combined: {D_combined}")
    return D_combined


def associate_track_pts_emb(detection_pts, detection_emb, trace_pts, tracker_emb, w_similarity: float = 0.5,
                            w_keypoints=0.5) -> np.ndarray:

    try:
        num_key_points = len(trace_pts)
        num_det_pts = len(detection_pts)

        similarity_matrix = cosine_similarity(tracker_emb, detection_emb)
        cost_similarity = 1 - similarity_matrix  # Convert similarity to cost

        cost_keypoints = cdist(trace_pts, detection_pts, metric="euclidean")

        combined_cost_matrix = w_similarity * cost_similarity + w_keypoints * cost_keypoints

        # Pad matrix for dummy assignments (if needed)
        if num_key_points > num_det_pts:
            padding = np.full((num_key_points, num_key_points - num_det_pts), 1e6)
            combined_cost_matrix = np.hstack((combined_cost_matrix, padding))
        elif num_det_pts > num_key_points:
            padding = np.full((num_det_pts - num_key_points, num_det_pts), 1e6)
            combined_cost_matrix = np.vstack((combined_cost_matrix, padding))

        # Compute optimal assignment using Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(combined_cost_matrix)
        assignments = [(row, col) for row, col in zip(row_ind, col_ind) if col < num_det_pts]

        for t_idx, d_idx in assignments:
            tracker_pts_pretty = ", ".join([f"{pt:.2f}" for pt in trace_pts[t_idx]])
            detection_pts_pretty = ", ".join([f"{pt:.2f}" for pt in detection_pts[d_idx]])
            info(f"Track point {t_idx} {tracker_pts_pretty}-> Detection point {d_idx} {detection_pts_pretty}with combined cost {combined_cost_matrix[t_idx, d_idx]:.2f}")

        return assignments, combined_cost_matrix
    except Exception as e:
        err(f"Error in associate_track_pts_emb: {e}")
        return [], []


def associate_trace_pts(detection_pts, trace_pts) -> np.ndarray:

    num_det_pts = len(detection_pts)

    try:
        cost_keypoints = cdist(trace_pts, detection_pts, metric="euclidean")

        # Compute optimal assignment using Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_keypoints)
        assignments = [(row, col) for row, col in zip(row_ind, col_ind) if col < num_det_pts]

        for t_idx, d_idx in assignments:
            detection_pts_pretty = ", ".join([f"{pt:.2f}" for pt in detection_pts[d_idx]])
            info(f"Track point {t_idx} {trace_pts[t_idx]}-> Detection point {d_idx} {detection_pts_pretty} with combined cost {cost_keypoints[t_idx, d_idx]:.2f}")

        return assignments, cost_keypoints
    except Exception as e:
        err(f"Error in associate_trace_pts: {e}")
        return [], []