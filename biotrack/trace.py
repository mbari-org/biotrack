# biotrack, Apache-2.0 license
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

    def get_box(self, frame_num: int):
        if frame_num in self.box.keys():
            return self.box[frame_num]
        return None

    def get_pt(self, frame_num: int):
        if frame_num in self.pt.keys():
            return self.pt[frame_num]
        if frame_num == -1 and len(self.pt) > 0:
            return list(self.pt.values())[-1]
        return None

    def close(self):
        self.closed = True

    def is_closed(self) -> bool:
        return self.closed

    def compute_acc(self) -> bool:
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
            info(f"{self.track_id}:{self.id} acceleration {np.round(acceleration_mag * 1000)}")
            return acceleration_mag
        return acceleration_mag

    @property
    def last_update_frame(self):
        frames = list(self.pt)
        if len(frames) > 0:
            return frames[-1]
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
        info(f"{self.track_id}:{self.trace_id} last_update_frame {self.last_update_frame} {pts_pretty}")

    def get_frames(self):
        return list(self.pt.keys())

    def get(self, frame_num: int, rescale=True) -> (np.array, str, np.array, float):
        if frame_num not in self.pt.keys():
            return None, None, None, 0.
        pt = self.pt[frame_num]
        # If there is a box in the frame, return it
        if self.box[frame_num] is not None:
            box = self.box[frame_num]
        else:
            box = []
        if rescale:
            pt, box = self.rescale(pt, box)
        if frame_num in self.score.keys():
            score = self.score[frame_num]
        else:
            score = 0.
        return pt, self.best_labels, box, score

    def predict(self) -> np.array:
        return list(self.pt.values())[-1]

    def update_pt(self, pt:np.array, frame_num: int):
        if self.is_closed():
            return

        info(f"{self.track_id}:{self.id} updating frame {frame_num} with points {pt}. No score/label")
        self.pt[frame_num] = pt

    def update_pt_box(self, label: str, pt: np.array, frame_num: int, box: np.array = None,
                      score: float = 0.) -> None:
        if self.is_closed():
            info(f"{self.track_id}:{self.id} is closed")
            return

        info(f"{self.track_id}:{self.id} updating frame {frame_num} with points {pt} {label} {score}")
        self.pt[frame_num] = pt
        self.label[frame_num] = label
        self.box[frame_num] = box
        self.score[frame_num] = score

        # Reduce the impact of the early detections by only considering the last 20 frames
        scores = np.array(list(self.score.values()))
        labels = list(self.label.values())
        if len(scores) > 30:
            scores = scores[-30:]
            labels = labels[-30:]

        # Calculate the weighted score for each label
        label_scores = defaultdict(float)
        for score, label in zip(scores, labels):
            label_scores[label] += float(score)

        label_counts = Counter(labels)

        # Normalize the scores by the number of occurrences of each label
        normalized_label_scores = {label: label_scores[label] / label_counts[label] for label in label_scores}

        # Sort the labels by normalized scores
        sorted_normalized_labels = sorted(normalized_label_scores.items(), key=lambda x: x[1], reverse=True)

        # Extract the top-1 and top-2 labels
        top_1_label, top_1_score = sorted_normalized_labels[0]
        if len(sorted_normalized_labels) > 1:  # Check if there's a second label
            top_2_label, top_2_score = sorted_normalized_labels[1]
        else:
            top_2_label, top_2_score = "marine organism", 0.  # Handle case with only one label

        self.winning_labels = [top_1_label, top_2_label]
        self.winning_scores = [top_1_score, top_2_score]
        pts_pretty = [f"{pt[0]:.2f},{pt[1]:.2f},{label},{score}" for pt, label, score in
                      zip(self.pt.values(), self.label.values(), self.score.values())]
        total_frames = len(self.pt)
        start_frame = list(self.pt)[0]
        info(f"Updating tracker {self.id} total_frames {total_frames} updated start {start_frame} to {frame_num} {pts_pretty} with label {self.winning_labels[0]}, score {self.winning_scores[0]}")
