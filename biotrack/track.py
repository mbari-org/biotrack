# biotrack, CC-BY-NC license
# Filename: biotrack/track.py
# Description:  Basic track object to contain and update_pt_box tracks
from collections import defaultdict
from biotrack.logger import info, err
from biotrack.trace import Trace
import numpy as np


class Track:
    NUM_KP = 4 # Number of key points to track

    def __init__(self, track_id: int, x_scale: float, y_scale: float, **kwargs):
        max_empty_frames = kwargs.get("max_empty_frames", 30)
        max_frames = kwargs.get("max_frames", 300)
        info(f"Creating tracker {track_id}. Max empty frame {max_empty_frames} Max frames {max_frames}")
        self.traces = [Trace(track_id, i) for i in range(Track.NUM_KP)]
        self.max_empty_frames = max_empty_frames
        self.max_frames = max_frames
        self.frames = defaultdict(int)
        self.id = track_id
        self.closed = False
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.emb = []

    @property
    def track_id(self):
        return self.id

    @property
    def num_frames(self):
        # Get the longest trace
        longest_trace = self.traces[0]
        for trace in self.traces:
            if trace.num_frames > longest_trace.num_frames:
                longest_trace = trace
        return longest_trace.num_frames

    def embedding(self):
        return self.emb

    def get_traces(self):
        return self.traces

    def is_closed(self) -> bool:
        return self.closed

    def dump(self):
        info(f"================Tracker {self.id} dump================")
        for trace in self.traces:
            trace.dump()
        info(f"=====================================================")

    def close_check(self, frame_num: int) -> bool:
        # Find the longest trace and get the average accuracy
        longest_trace = self.traces[0]
        for trace in self.traces:
            if trace.num_frames > longest_trace.num_frames:
                longest_trace = trace

        avg_acc, _ = longest_trace.compute_acc_vel()
        last_updated_frame = longest_trace.last_update_frame

        start_frame = self.traces[0].start_frame
        if frame_num < start_frame:
            return

        is_closed = ((frame_num - last_updated_frame + 1) >= self.max_empty_frames or
                     longest_trace.num_frames >= self.max_frames and avg_acc < 10. and avg_acc != -1)
        info(f"{self.id} is_closed {is_closed} frame_num {frame_num} last_updated_frame {last_updated_frame} "
             f"max_empty_frame {self.max_empty_frames} max_frames {self.max_frames}")
        self.closed = is_closed

    def rescale_pt(self, pt: np.array) -> np.array:
        pt_rescale = pt.copy()
        pt_rescale[0] = pt[0] * self.x_scale
        pt_rescale[1] = pt[1] * self.y_scale
        return pt_rescale

    def rescale_box(self, box: np.array) -> np.array:
        box_rescale = box.copy()
        box_rescale[0] = box[0] * self.x_scale
        box_rescale[1] = box[1] * self.y_scale
        box_rescale[2] = box[2] * self.x_scale
        box_rescale[3] = box[3] * self.y_scale
        return box_rescale

    def get_best(self, rescale=True) -> (int, np.array, str, np.array, float):
        # Get the longest trace
        longest_trace = self.traces[0]
        for trace in self.traces:
            if trace.num_frames > longest_trace.num_frames:
                longest_trace = trace

        frames = longest_trace.get_frames()
        if len(frames) == 0:
            return None, None, None, None, 0.

        # Find a frame somewhere near the middle of the trace that has a box
        possible_frames = frames[len(frames) // 2:]
        found = False
        for frame_num in possible_frames:
            best_box = longest_trace.get_box(frame_num)
            best_pt = longest_trace.get_pt(frame_num)
            best_labels = longest_trace.best_labels
            best_scores = longest_trace.best_scores
            best_frame = frame_num
            if best_box is not None:
                found = True
                break

        # Choose the first frame if no box is found
        if not found:
            best_frame = frames[0]
            best_box = longest_trace.get_box(best_frame)
            best_pt = longest_trace.get_pt(best_frame)
            best_labels = longest_trace.best_labels
            best_scores = longest_trace.best_scores

        if rescale:
            best_box = self.rescale_box(best_box)
            best_pt = self.rescale_pt(best_pt)
        return best_frame, best_pt, best_labels, best_box, best_scores

    def get(self, frame_num:int = -1, rescale=True) -> (int, np.array, str, np.array, float):
        if frame_num == -1:
            frame_num = list(self.pt.keys())[-3]

        pt = None
        for trace in self.traces:
            if frame_num in trace.pt.keys():
                pt = trace.get_pt(frame_num)
                break

        box = None
        for trace in self.traces:
            if frame_num in trace.box.keys():
                box = trace.get_box(frame_num)
                break

        if rescale:
            if box is not None:
                box = self.rescale_box(box)
            if pt is not None:
                pt = self.rescale_pt(pt)

        # If there is no point and no box, return None
        if pt is None and box is None:
            return None, None, None, 0.

        # Get the best score/label from the longest trace
        longest_trace = self.traces[0]
        for trace in self.traces:
            if trace.num_frames > longest_trace.num_frames:
                longest_trace = trace
        best_labels = longest_trace.best_labels
        best_scores = longest_trace.best_scores

        return pt, best_labels[0], box, best_scores[0]

    def predict(self) -> np.array:
        predictions = []
        for trace in self.traces:
            pt = trace.get_pt(-1)
            _, vel = trace.compute_acc_vel()
            if pt is not None:
                # Project the point to the next frame within the bounds of the image
                pt[0] = max(0, min(1.0, pt[0] + vel[0]))
                pt[1] = max(0, min(1.0, pt[1] + vel[1]))
                predictions.append(pt)
        return np.array(predictions)

    def init(self, trace_id:int, label: str, pt: np.array, emb: np.array, frame_num: int, box:np.array = None, score:float = None) -> None:
        if self.is_closed():
            info(f"{self.id} is closed")
            return

        self.traces[trace_id].update_pt_box(label, pt, frame_num, box, score)

        # Require an embedding to update_pt_box the tracker
        assert len(emb) > 0, "Embedding is required to initialize a tracker"
        self.emb = emb
        self.close_check(frame_num)