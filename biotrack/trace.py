# biotrack, CC-BY-NC license
# Filename: biotrack/trace.py
# Description:  Basic trace object to contain and update_pt_box traces
from collections import defaultdict, Counter
from biotrack.logger import info, debug
import numpy as np


class Trace:

    def __init__(self, track_id:int, trace_id: int):
        info(f"Creating trace {trace_id} associated with track {track_id} ")
        self.trace_id = trace_id
        self.track_id = track_id
        self.pt = {}
        self.label = {}
        self.score = {}
        self.box = {}
        self.closed = False
        self.winning_labels = ["marine organism", "marine organism"]
        self.winning_scores = [0., 0.]

    @property
    def start_frame(self):
        frames = list(self.pt)
        if len(frames) > 0:
            return frames[0]
        return -1

    @property
    def id(self):
        return self.trace_id

    @property
    def embedding(self):
        return self.emb

    @property
    def best_labels(self):
        return self.winning_labels

    @property
    def best_scores(self):
        return self.winning_scores

    def get_box(self, frame_num: int, rescale=False):
        if frame_num in self.box.keys():
            if rescale:
                return self.rescale(self.box[frame_num])
            return self.box[frame_num]
        return None

    def get_pt(self, frame_num: int, rescale=False):
        if frame_num in self.pt.keys():
            if rescale:
                return self.rescale(self.pt[frame_num])
            return self.pt[frame_num]
        if frame_num == -1 and len(self.pt) > 0:
            if rescale:
                return self.rescale(list(self.pt.values())[-1])
            return list(self.pt.values())[-1]
        return None

    def close(self):
        self.closed = True

    def is_closed(self) -> bool:
        return self.closed

    def compute_acc_vel(self) -> float:
        # If the acceleration is very low, then the object being tracked is likely to be stationary
        # allow the track to remain open for a longer period of time if it is
        pt = np.array([pt for pt in self.pt.values()])
        frames = np.array([int(frame) for frame in self.pt.keys()])
        acceleration_mag = -1
        if len(pt) > 2:
            delta_pos = np.diff(pt, axis=0)
            delta_time = np.diff(frames)
            velocity = delta_pos / delta_time[:, np.newaxis]
            acceleration = np.diff(velocity, axis=0) / delta_time[1:, np.newaxis]
            acceleration_mag = np.linalg.norm(acceleration[-1])  # Get the last acceleration
            info(f"{self.track_id}:{self.id} acceleration {np.round(acceleration_mag * 1000)}, velocity {np.round(np.linalg.norm(velocity[-1]) * 1000)}")
            return acceleration_mag, velocity[-1]
        return acceleration_mag, np.array([0., 0.])

    @property
    def last_update_frame(self):
        frames = list(self.pt)
        if len(frames) > 0:
            return frames[-1]
        return -1

    @property
    def start_frame(self):
        frames = list(self.pt)
        if len(frames) > 0:
            return frames[0]
        return -1

    @property
    def num_frames(self):
        frames = list(self.pt)
        if len(frames) > 0:
            return frames[-1] - frames[0] + 1
        return 0

    def dump(self):
        pts_pretty = [f"{pt[0]:.2f},{pt[1]:.2f},{label},{score}" for pt, label, score in
                      zip(self.pt.values(), self.label.values(), self.score.values())]
        info(f"{self.track_id}:{self.trace_id} start_frame {self.start_frame} last_update_frame {self.last_update_frame} {pts_pretty}")

    def get_frames(self):
        return list(self.pt.keys())

    def predict(self) -> np.array:
        return list(self.pt.values())[-1]

    def update_pt(self, pt:np.array, frame_num: int):
        if self.is_closed() or frame_num < self.last_update_frame:
            return

        info(f"{self.track_id}:{self.id} updating frame {frame_num} with points {pt}. No score/label")
        self.pt[frame_num] = pt
        self.score[frame_num] = 0.
        self.label[frame_num] = "marine organism"

    def update_pt_box(self, label: str, pt: np.array, frame_num: int, box: np.array = None,
                      score: float = 0.) -> None:
        if self.is_closed() or frame_num < self.last_update_frame:
            return

        info(f"{self.track_id}:{self.id} updating frame {frame_num} with points {pt} {label} {score}")
        self.pt[frame_num] = pt
        self.label[frame_num] = label
        self.box[frame_num] = box
        self.score[frame_num] = score

        scores = np.array(list(self.score.values()))
        labels = list(self.label.values())

        # Sort the labels by the maximum score and get the top-1 and top-2 labels
        sorted_labels = sorted(zip(labels, scores), key=lambda x: x[1], reverse=True)
        (top_1_label, top_1_score) = sorted_labels[0]
        if len(sorted_labels) > 1:
            (top_2_label, top_2_score) = sorted_labels[1]
        else:
            top_2_label, top_2_score = "marine organism", 0.

        self.winning_labels = [top_1_label, top_2_label]
        self.winning_scores = [top_1_score, top_2_score]
        pts_pretty = [f"{pt[0]:.2f},{pt[1]:.2f},{label},{score}" for pt, label, score in
                      zip(self.pt.values(), self.label.values(), self.score.values())]
        total_frames = len(self.pt)
        start_frame = list(self.pt)[0]
        info(f"Updating tracker {self.id} total_frames {total_frames} updated start {start_frame} to {frame_num} {pts_pretty} with label {self.winning_labels[0]}, score {self.winning_scores[0]}")
