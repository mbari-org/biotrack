# biotrack, Apache-2.0 license
# Filename: biotrack/tracker/track.py
# Description:  Basic track object to contain and update tracks
from collections import Counter

from biotrack.logger import info, debug
import numpy as np


class Track:
    def __init__(self, track_id: int, label: str, pt: np.array, emb: np.array, frame: int, x_scale: float, y_scale: float, box: np.array = None, score: float = 0., **kwargs):
        max_empty_frames = kwargs.get("max_empty_frames", 30)
        max_frames = kwargs.get("max_frames", 300)
        info(f"Creating tracker {track_id} at {frame}:{pt},{score}. Max empty frame {max_empty_frames} Max frames {max_frames}")
        self.max_empty_frames = max_empty_frames
        self.max_frames = max_frames
        self.id = track_id
        self.pt = {frame: pt}
        self.label = {frame: label}
        self.score = {frame: score}
        self.box = {frame: box}
        self.emb = emb
        self.best_label = label
        self.best_score = score
        self.start_frame = frame
        self.last_updated_frame = frame
        self.x_scale = x_scale
        self.y_scale = y_scale

    @property
    def track_id(self):
        return self.id

    @property
    def embedding(self):
        return self.emb

    def is_closed(self, frame_num: int) -> bool:
        is_closed = (frame_num - self.last_updated_frame + 1) >= self.max_empty_frames or len(self.pt) >= self.max_frames
        info(f"Tracker {self.id} is_closed {is_closed} frame_num {frame_num} last_updated_frame {self.last_updated_frame} max_empty_frame {self.max_empty_frames} max_frames {self.max_frames}")
        return is_closed

    @property
    def last_update_frame(self):
        return self.last_updated_frame

    def rescale(self, pt: np.array, box: np.array) -> (np.array, np.array):
        pt_rescale = pt.copy()
        pt_rescale[0] = pt[0] * self.x_scale
        pt_rescale[1] = pt[1] * self.y_scale
        if box is not None:
            box_rescale = box.copy()
            box_rescale[0] = box[0] * self.x_scale
            box_rescale[1] = box[1] * self.y_scale
            box_rescale[2] = box[2] * self.x_scale
            box_rescale[3] = box[3] * self.y_scale
        else:
            box_rescale = box
        return pt_rescale, box_rescale

    def get_best(self, rescale=True) -> (int, np.array, str, np.array, float):
        # Get the best box which is a few frames behind the last_updated_frame
        # This is pretty arbitrary, but sometimes the last box is too blurry or not visible
        num_frames = len(self.pt.keys())
        if num_frames > 3:
            frame_num = list(self.pt.keys())[-3]
            box = self.box[frame_num]
            pt = self.pt[frame_num]
        else: # Handle the case where there is only one frame tracked
            frame_num = self.last_updated_frame
            box = self.box[frame_num]
            pt = self.pt[frame_num]
        if rescale:
            pt, box = self.rescale(pt, box)
        return frame_num, pt, self.best_label, box, self.best_score

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
        return pt, self.best_label, box, score

    def predict(self) -> np.array:
        return self.pt[self.last_updated_frame]

    def update(self, label: str, pt: np.array, emb: np.array, frame_num: int, box:np.array = None, score:float = None) -> None:
        if self.is_closed(frame_num):
            debug(f"Tracker {self.id} has a gap from {self.last_updated_frame} to {frame_num} or more than max_frames {self.max_frames}")
            return

        # If updating the same last_updated_frame, replace the point
        if frame_num == self.last_updated_frame:
            info(f"Updating tracker {self.id} at frame {frame_num} with point {pt}")
            self.pt[frame_num] = pt
            self.label[frame_num] = label
            self.box[frame_num] = box
            self.score[frame_num] = score
            # If there is a valid embedding, update it
            if len(emb) > 0:
                self.emb = emb
            return

        # If adding in a new last_updated_frame, add the point
        self.pt[frame_num] = pt
        self.label[frame_num] = label
        self.box[frame_num] = box
        self.score[frame_num] = score
        if len(emb) > 0:
            self.emb = emb
        self.last_updated_frame = frame_num

        # Update the best_label with that which occurs the most that has a score > 0. This is a simple majority vote
        data = [(pred, score) for pred, score in zip(self.label.values(), self.score.values()) if float(score) > 0.]

        if len(data) > 0:
            p, s = zip(*data)
            model_predictions = list(p)
            model_scores = list(s)

            # Count occurrences of each prediction in the top lists
            counter = Counter(model_predictions)

            majority_count = (len(data) // 2) + 1

            majority_predictions = [pred for pred, count in counter.items() if count >= majority_count]

            # If there are no majority predictions
            if len(majority_predictions) == 0:
                # Pick the prediction with the highest score
                # best_pred, max_score = max_score_p(model_predictions, model_scores)
                self.best_label = "marine organism"
                self.best_score = 0.0
            else:
                self.best_label = majority_predictions[0]
                best_score = 0.0
                num_majority = 0
                # Sum all the scores for the majority predictions
                for pred, score in data:
                    if pred in majority_predictions:
                        best_score += float(score)
                        num_majority += 1
                self.best_score /= num_majority
        else:
            self.best_label = "marine organism"
            self.best_score = 0.0

        pts_pretty = [f"{pt[0]:.2f},{pt[1]:.2f},{label},{score}" for pt, label, score in zip(self.pt.values(), self.label.values(), self.score.values())]
        total_frames = len(self.pt)
        info(f"Updating tracker {self.id} total_frames {total_frames} updated start {self.start_frame} to {self.last_updated_frame} {pts_pretty} with label {self.best_label}, score {self.best_score}")
