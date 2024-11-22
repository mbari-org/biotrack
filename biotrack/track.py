# biotrack, Apache-2.0 license
# Filename: biotrack/track.py
# Description:  Basic track object to contain and update tracks
from collections import Counter

from biotrack.logger import info, debug
import numpy as np


def max_score_p(model_predictions, model_scores):
    """Find the top prediction"""
    max_score = 0.0

    for row in zip(model_predictions, model_scores):
        prediction, score = row
        score = float(score)
        if score > max_score:
            max_score = score
            best_pred = prediction

    return best_pred, max_score


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
        self.closed = False
        self.x_scale = x_scale
        self.y_scale = y_scale

    @property
    def track_id(self):
        return self.id

    @property
    def embedding(self):
        return self.emb

    def is_closed(self) -> bool:
        return self.closed

    def close(self, frame_num: int) -> bool:
        # If the acceleration is very low, then the object being tracked is likely to be stationary
        # allow the track to remain open for a longer period of time if it is
        pts = np.array([pt for pt in self.pt.values()])
        frames = np.array([int(frame) for frame in self.pt.keys()])
        acceleration_mag = 100.
        if len(pts) > 2:
            delta_pos = np.diff(pts, axis=0)
            delta_time = np.diff(frames)
            velocity = delta_pos / delta_time[:, np.newaxis]
            acceleration = np.diff(velocity, axis=0) / delta_time[1:, np.newaxis]
            acceleration_mag = np.linalg.norm(acceleration[-1]) # Get the last acceleration
            info(f"Tracker {self.id} acceleration {np.round(acceleration_mag*1000)}")
        is_closed = ((frame_num - self.last_updated_frame + 1) >= self.max_empty_frames or
                     len(self.pt) >= self.max_frames and acceleration_mag < 3.)
        info(f"Tracker {self.id} is_closed {is_closed} frame_num {frame_num} last_updated_frame {self.last_updated_frame} "
             f"max_empty_frame {self.max_empty_frames} max_frames {self.max_frames}")
        self.closed = is_closed

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
        if self.is_closed():
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

        # Reduce the impact of the early detections by only considering the last 10 frames
        scores = np.array(list(self.score.values()))
        labels = list(self.label.values())
        if len(scores) > 10:
            scores = scores[-10:]
            labels = labels[-10:]

        # Update the best_label with that of the highest scoring recent label
        max_score = max(scores)
        max_frame = np.argmax(scores)
        self.best_label = labels[max_frame]
        self.best_score = max_score

        pts_pretty = [f"{pt[0]:.2f},{pt[1]:.2f},{label},{score}" for pt, label, score in zip(self.pt.values(), self.label.values(), self.score.values())]
        total_frames = len(self.pt)
        info(f"Updating tracker {self.id} total_frames {total_frames} updated start {self.start_frame} to {self.last_updated_frame} {pts_pretty} with label {self.best_label}, score {self.best_score}")
