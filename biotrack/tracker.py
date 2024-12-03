# biotrack, Apache-2.0 license
# Filename: biotrack/tracker/tracker.py
# Description:  Main tracker class for tracking objects with points, label and embeddings using the CoTracker model and BioClip ViT embeddings
from dbm.dumb import error

import cv2
from pathlib import Path

from PIL import Image

import torch
import piexif
from typing import List, Dict, Tuple
import numpy as np
from cotracker.predictor import CoTrackerPredictor
from biotrack.assoc import associate_track_pts_emb, associate_trace_pts
from biotrack.embedding import ViTWrapper
from biotrack.track import Track
from biotrack.trace import Trace
from biotrack.logger import create_logger_file, info, debug

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class BioTracker:
    def __init__(self, image_width: int, image_height: int, device_id: int = 0):
        self.logger = create_logger_file()
        self.image_width = image_width
        self.image_height = image_height
        self.model_width = 640
        self.model_height = 360
        self.image_width = image_width
        self.image_height = image_height
        self.open_trackers: List[Track] = []
        self.closed_trackers: List[Track] = []
        self.next_track_id = 0  # Unique ID for new tracks
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
        self.offline_model = CoTrackerPredictor(checkpoint=None)
        self.offline_model.model = model.model
        self.offline_model.step = model.model.window_len // 2

        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        self.offline_model.model.to(self.device)

        # Initialize the model for computing crop embeddings
        self.vit_wrapper = ViTWrapper(DEFAULT_DEVICE, device_id)

    def update_trackers_queries(self, frame_num: int, points: np.array, labels: List[str], scores: np.array, boxes: np.array,  d_emb: np.array, **kwargs):
        """
        Update the tracker with the queries. This will seed new tracks if they cannot be associated with existing tracks
        :param boxes:  the bounding boxes of the queries in the format [[x1,y1,x2,y2],[x1,y1,x2,y2]...] in the normalized scale 0-1
        :param scores:  the scores of the queries in the format [score1, score2, score3...] in the normalized scale 0-1
        :param labels:  the labels of the queries in the format [label1, label2, label3...]
        :param frame_num:  the starting frame number of the batch
        :param points:  points in the format [[x1,y1],[x2,y2],[[x1,y1],[x2,y2],[x3,y3...] in the normalized scale 0-1
        :param kwargs:
        :return:
        """
        if len(points) == 0:
            return

        info(f"Query updating tracks with {len(points)} query points in frame {frame_num}")
        max_cost = kwargs.get("max_cost", 0.4)

        t_pts, t_emb = self.get_tracks_detail()

        debug(f"Associating {len(points)} points with {len(t_pts)} traces")

        # If there are no open traces, cannot associate the queries with existing traces so create new ones
        if len(t_pts) == 0:
            num_tracks = len(points) // Track.NUM_KP
            # Every NUM_KP points is a new track
            for i in range(len(points)):
                if i % Track.NUM_KP == 0:
                    info(f"Creating new track {self.next_track_id} at {frame_num}")
                    self.open_trackers.append(Track(self.next_track_id, self.image_width, self.image_height, **kwargs))
                    self.next_track_id += 1

            # Get the newly created tracks and initialize the traces
            open_tracks = self.open_trackers[-num_tracks:]
            for i, pt in enumerate(points):
                j = i // Track.NUM_KP
                trace_id = i % Track.NUM_KP
                emb = d_emb[j]
                box = boxes[j]
                label = labels[j]
                score = scores[j]
                open_tracks[j].init(trace_id, label, pt, emb, frame_num, box=box, score=score)
            return
        else:
            # Associate the new points with the existing track traces
            assignment, costs = associate_track_pts_emb(detection_pts=points, detection_emb=d_emb, tracker_pts=t_pts, tracker_emb=t_emb)
            if len(assignment) == 0:
                return
            open_traces = [t.get_traces() for t in self.open_trackers]
            open_traces = [item for sublist in open_traces for item in sublist]

            unassigned = []
            for t_idx, d_idx in assignment:
                cost = costs[t_idx, d_idx]
                if cost < max_cost:
                    box = boxes[d_idx // Track.NUM_KP ]
                    label = labels[d_idx // Track.NUM_KP]
                    score = scores[d_idx // Track.NUM_KP]
                    open_traces[t_idx].update_pt_box(label, points[d_idx], frame_num, box, score)
                else:
                    debug(f"Track trace {t_idx} not associated with detection point {d_idx} with cost {cost:.2f}. Cost threshold {max_cost:.2f}")
                    unassigned.append(d_idx)

            # Create new tracks for the unassigned points.  There are NUM_KP points per track and unassigned should be multiples of NUM_KP
            if len(unassigned) % Track.NUM_KP != 0:
                error(f"Unassigned points {len(unassigned)} is not a multiple of {Track.NUM_KP}")

            for i in range(0, len(unassigned), Track.NUM_KP):
                info(f"Creating new track {self.next_track_id} at {frame_num}")
                track = Track(self.next_track_id, self.image_width, self.image_height, **kwargs)
                unassigned_idx = unassigned[i:i + Track.NUM_KP]
                for j, d_idx in enumerate(unassigned_idx):
                    box = boxes[d_idx // Track.NUM_KP]
                    label = labels[d_idx // Track.NUM_KP]
                    score = scores[d_idx // Track.NUM_KP]
                    track.init(j, label, points[d_idx], d_emb[d_idx // Track.NUM_KP], frame_num, box=box, score=score)
                self.open_trackers.append(track)
                self.next_track_id += 1


    def get_tracks_detail(self):
        # Get predicted track traces and embeddings from existing trackers
        open_tracks = [t for t in self.open_trackers if not t.is_closed()]
        t_pts = np.zeros((len(open_tracks)*Track.NUM_KP, 2))
        t_emb = np.zeros((len(open_tracks)*Track.NUM_KP, self.vit_wrapper.vector_dimensions))
        j = 0
        for i, t in enumerate(open_tracks):
            predicted_pts = t.predict()
            predicted_emb = t.embedding()
            if len(predicted_pts) == 0:
                continue
            # There are NUM_KP points per tracker, so we need to add the same embedding for each point
            for pt in predicted_pts:
                t_pts[j] = pt
                t_emb[j] = predicted_emb
                j += 1
        return t_pts, t_emb


    def update_trackers_pred(self, frame_num: int, points: np.array, **kwargs):
        """
        Update the tracker with new det_query and crops
        :param frame_num: the starting frame number of the batch
        :param points: points in the format [[x1,y1],[x2,y2],[[x1,y1],[x2,y2],[x3,y3...] in the normalized scale 0-1
        a collection of det_query detected in each last_updated_frame.
        :param kwargs:
        :return:
        """
        info(f"Prediction update_pt_box tracks with {len(points)} points in frames batch starting at {frame_num}")

        max_cost = kwargs.get("max_cost", 0.5)

        # If there are no tracked points
        if len(points) == 0:
            return

        t_pts, _ = self.get_tracks_detail()
        debug("Track points:")
        for pt in t_pts:
            debug(f"{pt}")

        debug(f"Associating {len(points)} points with {len(t_pts)} traces")

        if len(t_pts) == 0: # no open traces to associate with
            return

        # Associate the new points with the existing track traces
        assign, costs = associate_trace_pts(detection_pts=points, trace_pts=t_pts)
        open_traces = [t.get_traces() for t in self.open_trackers][0]

        for t_idx, d_idx in assign:
            cost = costs[t_idx, d_idx]
            if t_idx < len(open_traces) and cost < max_cost:
                open_traces[t_idx].update_pt(points[d_idx], frame_num)
            # else:
            #     info(f"Track trace {t_idx} not associated with detection point {d_idx} with cost {cost:.2f}")
            #     info(f"Creating new track {self.next_track_id} at {frame_num}")
            #     track = Track(self.next_track_id, self.image_width, self.image_height, **kwargs)
            #     track.init(0, "Unknown", points[d_idx], [], frame_num)
            #     self.open_trackers.append(track)
            #     open_traces = [t.get_traces() for t in self.open_trackers][0]
            #     self.next_track_id += 1

    def check(self, frame_num: int):
        i = len(self.open_trackers)
        for t in reversed(self.open_trackers):
            i -= 1
            t.close_check(frame_num)
            t.dump()
            if t.is_closed():
                info(f"Closing track {t.track_id}")
                self.closed_trackers.append(t)
                self.open_trackers.pop(i)

    def update_batch(self, frame_range: Tuple[int, int], frames: np.ndarray, detections: Dict, **kwargs):
        """
        Update the tracker with new frames and det_query
        :param frame_range: a tuple of the starting and ending frame numbers
        :param frames: numpy array of frames in the format [frame1, frame2, frame3...]
        :param detections: dictionary of det_query in the format {["x": x, "y": y, "xx": x, "xy": xy, "crop_path": crop_path, "frame": frame, "class_name": class_name, "score": score]}
        :param kwargs:
        :return:
        """
        det_query = {}
        crop_query = {}
        num_frames = len(frames)
        for d in detections:
            if not Path(d["crop_path"]).exists():
                continue

            # Skip queries that are outside the frame range - must define queries in the same frame relative index
            if d['frame'] - frame_range[0] >= num_frames:
                continue

            image = cv2.imread(d["crop_path"])
            gray_crop = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Choose the best keypoint to track based on SIFT
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(gray_crop, None)
            keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)
            if len(keypoints) == 0:
                # Choose the NUM_KP keypoints in a grid pattern if no keypoints are found
                top_kps = []
                for i in range(Track.NUM_KP):
                    x = (i % 3) * (image.shape[1] // 3)
                    y = (i // 3) * (image.shape[0] // 3)
                    top_kps.append([x, y])

            # # Choose the top NUM_KP keypoints by response
            top_kps = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints[:Track.NUM_KP]])

            # Extract originating bounding box from EXIF UserComment tag
            # This is the box the crop was extracted from in the raw image
            bbox = None
            with Image.open(d["crop_path"]) as img:
                exif_dict = img._getexif()
                if exif_dict is not None:
                    user_comment = exif_dict[piexif.ExifIFD.UserComment]
                    user_comment = user_comment.decode("utf-8")
                    if user_comment.startswith("bbox:"):
                        bbox_str = user_comment.split("bbox:")[1]
                        bbox = eval(bbox_str)

            # Should never get here unless something is wrong
            if bbox is None:
                raise ValueError(f"No bounding box found in {d['crop_path']}")

            # Actual size of the crop which is rescaled to 224x224 - this is used to find
            # the top left x, y in the original image for the query
            scale_x = (bbox[2] - bbox[0]) / 224
            scale_y = (bbox[3] - bbox[1]) / 224
            norm_kpts = np.zeros((len(top_kps), 2))
            for i, pt in enumerate(top_kps):
                x, y = pt
                x = x * scale_x
                y = y * scale_y
                x += bbox[0]
                y += bbox[1]
                # Adjust to 0-1 scale
                x /= self.image_width
                y /= self.image_height
                norm_kpts[i] = np.array([x,y])

            frame_num = d["frame"]
            bbox = np.array([bbox[0] / self.image_width, bbox[1] / self.image_height, bbox[2] / self.image_width, bbox[3] / self.image_height])

            # Add the best keypoints to the query
            info(f"Adding query for {norm_kpts} in frame {frame_num}")
            if frame_num not in det_query:
                det_query[frame_num] = []
                crop_query[frame_num] = []
            det_query[frame_num].append([norm_kpts, bbox])
            crop_query[frame_num].append(d["crop_path"])

        return self._update_batch(frame_range, frames, det_query, crop_query, **kwargs)

    def _update_batch(self, frame_range: Tuple[int, int], frames: np.ndarray, det_query: Dict, crop_query: Dict, **kwargs):
        """
        Update the tracker with the new frames, detections and crops of the detections
        :param frame_range: a tuple of the starting and ending frame numbers
        :param frames: numpy array of frames in the format [frame1, frame2, frame3...]
        :param det_query: a dictionary of detection keypoints, labels and boxes in the format {frame_num: [[[x1,y1],[x2,y2],...label,score,bbox],[[x1,y1],[x2,y2],label,score,bbox]...]} in the normalized scale 0-1
        :param crop_query: a dictionary of crop paths in the format {frame_num: [crop_path1, crop_path2, crop_path3...]}
        :param kwargs:
        :return:
        """
        if len(det_query) == 0 or len(crop_query) == 0:
            info("No data for frame")
            return []

        if len(det_query) != len(crop_query):
            info(f"Number of det_query {len(det_query)} and crop paths {len(crop_query)} do not match")
            return []

        # Compute the embeddings for the new query detection crops
        # Format the queries for the model, each query is [frame_number, x, y]
        q_emb = {}
        queries = []
        labels = {}
        scores = {}
        for f, det in det_query.items():
            # TODO: replace with parallel processing
            info(f"Computing embeddings for frame {f} {crop_query[f]}")
            batch_embeddings, predicted_classes, predicted_scores = self.vit_wrapper.process_images(crop_query[f])
            q_emb[f] = batch_embeddings
            labels[f] = predicted_classes
            scores[f] = predicted_scores
            for pt in det[0][0]:
                queries.append([f - frame_range[0], pt[0], pt[1]])

        if len(queries) == 0:
            info("No queries found")
            return []

        # Update with the queries - these seed new tracks and update_pt_box existing tracks
        for f in range(frame_range[0], frame_range[1]):
            if f not in det_query.keys():
                continue

            labels_in_frame = [l[0] for l in labels[f]] # Get the top label and score
            scores_in_frame = [s[0] for s in scores[f]]
            boxes_in_frame = [d[1] for d in det_query[f]]
            queries_in_frame = [d[:1] for d in det_query[f]]
            debug(f"Updating with queries {queries_in_frame} in frame {f}")
            self.update_trackers_queries(f, queries_in_frame[0][0], labels_in_frame, scores_in_frame, boxes_in_frame, q_emb[f], **kwargs)
            self.check(f)

        # Put the queries and frames into tensors and run the model with the backward tracking option which is
        # more accurate than the forward/online only tracking
        # Convert the queries to the model scale
        queries = np.array(queries)
        queries[:, 1] *= self.model_width
        queries[:, 2] *= self.model_height
        queries_t = torch.tensor(queries, dtype=torch.float32)
        frames = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2)
        # Put the data on the cuda device
        queries_t = queries_t.to(self.device)
        frames = frames.to(self.device)

        video_chunk = frames.unsqueeze(0)  # Shape: (B, T, C, H, W)
        info(f"Running co-tracker model with {len(queries)} queries frames {frame_range}")
        pred_pts, pred_visibilities = self.offline_model(video_chunk, queries=queries_t[None], backward_tracking=True)
        pred_pts, pred_visibilities = pred_pts.cpu().numpy(), pred_visibilities.cpu().numpy()
        # Convert the predictions back to the normalized scale
        pred_pts[:, :, :, 0] /= self.model_width
        pred_pts[:, :, :, 1] /= self.model_height

        # Update with predictions
        for f in range(frame_range[0], frame_range[1], 1):

            if len(pred_pts) == 0:
                debug(f"No predictions for frame {f}")
                continue
            pts = pred_pts[:, f - frame_range[0], :, :]
            if len(pts) == 0:
                debug(f"No predictions for frame {f}")
                continue

            # Filter out the det_query that are not visible
            pred_visibilities_in_frame = pred_visibilities[:, f - frame_range[0], :]
            debug(f"Found {len(pred_visibilities_in_frame[0])} predictions that are visible in frame {f}")
            filtered_pts = pts[pred_visibilities_in_frame]
            debug(f"{len(filtered_pts)} predictions are visible in {f}")
            for pt in filtered_pts:
                debug(f"{pt}")

            self.update_trackers_pred(f, filtered_pts, **kwargs)
            self.check(f)

        return self.open_trackers + self.closed_trackers

    def purge_closed_tracks(self):
        """
        Purge the closed tracks that are no longer needed
        :param frame_num: the current frame number
        :return:
        """
        # Close any tracks ready to close_check
        i = len(self.closed_trackers)
        for t in reversed(self.closed_trackers):
            i -= 1
            if t.is_closed():
                info(f"Removing closed track {t.track_id}")
                self.closed_trackers.pop(i)


    def get_tracks(self):
        """
        Get the open and closed tracks
        :return: a list of open and closed tracks
        """
        return self.open_trackers + self.closed_trackers