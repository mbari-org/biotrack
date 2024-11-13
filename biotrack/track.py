# biotrack, Apache-2.0 license
# Filename: biotrack/tracker/track.py
# Description:  Basic track object to contain and update tracks
from biotrack.logger import info, debug
import numpy as np


class Track:
    def __init__(self, label: str, pt: np.array, emb: np.array, frame: int, x_scale: float, y_scale: float, max_empty_frame: int = 5, max_frames: int = 60, id: int = 0):
        info(f"Creating tracker {id} at frame {frame} with point {pt} and emb {emb.shape}. Max empty frame {max_empty_frame} Max frames {max_frames}")
        self.max_empty_frame = max_empty_frame
        self.max_frames = max_frames
        self.id = id
        self.pt = {frame: pt}
        self.label = {frame: label}
        self.emb = emb
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
        is_closed = (frame_num - self.last_updated_frame + 1) >= self.max_empty_frame or len(self.pt) >= self.max_frames
        debug(f"Tracker {self.id} is_closed {is_closed} frame_num {frame_num} last_updated_frame {self.last_updated_frame} max_empty_frame {self.max_empty_frame} max_frames {self.max_frames}")
        return is_closed

    @property
    def last_update_frame(self):
        return self.last_updated_frame

    def get_point_label(self, frame_num: int, rescale=True) -> (np.array, str):
        if frame_num not in self.pt.keys():
            return None, None
        pt = self.pt[frame_num]
        label = self.label[frame_num]
        if rescale:
            pt[0] *= self.x_scale
            pt[1] *= self.y_scale
        return pt, label

    def predict(self) -> np.array:
        return self.pt[self.last_updated_frame]

    def update(self, label: str, pt: np.array, emb: np.array, frame_num: int) -> None:
        if self.is_closed(frame_num):
            debug(f"Tracker {self.id} has a gap from {self.last_updated_frame} to {frame_num} or more than max_frames {self.max_frames}")
            return

        # If updating the same last_updated_frame, replace the point
        if frame_num == self.last_updated_frame:
            info(f"Updating tracker {self.id} at frame {frame_num} with point {pt}")
            self.pt[frame_num] = pt
            self.label[frame_num] = label
            # If there is a valid embedding, update it
            if len(emb) > 0:
                self.emb = emb
            return

        # If adding in a new last_updated_frame, add the point
        self.pt[frame_num] = pt
        self.label[frame_num] = label
        if len(emb) > 0:
            self.emb = emb
        self.last_updated_frame = frame_num
        pts_pretty = [f"{pt[0]:.2f},{pt[1]:.2f},{label}" for pt, label in zip(self.pt.values(), self.label.values())]
        best_label = max(set(self.label.values()), key=list(self.label.values()).count)
        total_frames = len(self.pt)
        info(f"Updating tracker {self.id} total_frames {total_frames} updated start {self.start_frame} to {self.last_updated_frame} {pts_pretty} with label {best_label}")
