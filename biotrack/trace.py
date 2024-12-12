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
        self.label1 = {}
        self.label2 = {}
        self.score1 = {}
        self.score2 = {}
        self.box = {}
        self.coverage = {}
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
        # Display the last 10 trace points as a string
        labels1 = list(self.label1.values())[0:10]
        scores1 = list(self.score1.values())[0:10]
        coverage = list(self.coverage.values())[0:10]
        pts_str = ' ' .join(f"{pt[0]:.2f},{pt[1]:.2f},{label},{score:.2f},{coverage:.2f}" for pt, label, score, coverage in
                      zip(self.pt.values(), labels1, scores1, coverage))
        info(f"{self.track_id}:{self.trace_id} start_frame {self.start_frame} last_update_frame {self.last_update_frame} {pts_str}")

    def get_frames(self):
        return list(self.pt.keys())

    def predict(self) -> np.array:
        return list(self.pt.values())[-1]

    def update_pt(self, pt:np.array, frame_num: int):
        if self.is_closed() or frame_num < self.last_update_frame:
            return

        info(f"{self.track_id}:{self.id} updating frame {frame_num} with points {pt}. No score/label")
        self.pt[frame_num] = pt
        self.score1[frame_num] = 0.
        self.score2[frame_num] = 0.
        self.label1[frame_num] = "marine organism"
        self.label2[frame_num] = "marine organism"

    def update_pt_box(self, labels: [str], pt: np.array, frame_num: int, box: np.array = None,
                      scores: float =  [0.,0.], coverage: float =  0.) -> None:
        if self.is_closed() or frame_num < self.last_update_frame:
            return

        score_str = ",".join([f"{score:.2f}" for score in scores])
        info(f"{self.track_id}:{self.id} updating frame {frame_num} with points {pt} {labels} {score_str} {coverage:.2f}")
        self.pt[frame_num] = pt
        self.label1[frame_num] = labels[0]
        self.label2[frame_num] = labels[1]
        self.box[frame_num] = box
        self.score1[frame_num] = scores[0]
        self.score2[frame_num] = scores[1]

        labels1 = list(self.label1.values())
        labels2 = list(self.label2.values())
        scores1 = np.array(list(self.score1.values()))
        scores2 = np.array(list(self.score2.values()))

        # Sort the labels by the maximum score and get the top-1 label
        sorted_labels = sorted(zip(labels1, scores1), key=lambda x: x[1], reverse=True)
        (top_1_label, top_1_score) = sorted_labels[0]

        # Sort the labels by the maximum score and get the top-1 label
        sorted_labels = sorted(zip(labels2, scores2), key=lambda x: x[1], reverse=True)
        (top_2_label, top_2_score) = sorted_labels[0]

        self.winning_labels = [top_1_label, top_2_label]
        self.winning_scores = [top_1_score, top_2_score]
        pts_str = [f"{pt[0]:.2f},{pt[1]:.2f},{label},{score:.2f},{coverage:.2f}" for pt, label, score, coverage in
                      zip(self.pt.values(), self.label1.values(), self.score1.values(), self.coverage.values())]
        total_frames = len(self.pt)
        start_frame = list(self.pt)[0]
        info(f"Updating tracker {self.id} total_frames {total_frames} updated start {start_frame} to {frame_num} {pts_str} with label {self.winning_labels[0]}, score {self.winning_scores[0]}")
