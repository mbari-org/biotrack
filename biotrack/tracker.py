# biotrack, Apache-2.0 license
# Filename: biotrack/tracker/tracker.py
# Description:  Main tracker class for tracking objects with points, label and embeddings using the CoTracker model and BioClip ViT embeddings
import cv2
from pathlib import Path

from PIL import Image

import torch
import piexif
from typing import List, Dict, Tuple
import numpy as np
from cotracker.predictor import CoTrackerPredictor
from biotrack.assoc import associate
from biotrack.embedding import ViTWrapper, compute_embedding_vits
from biotrack.track import Track
from biotrack.logger import create_logger_file, info, debug

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class BioTracker:
    def __init__(self, image_width: int, image_height: int):
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
        if DEFAULT_DEVICE == "cuda":
            self.offline_model.model = self.offline_model.model.cuda()

        # Initialize the model for computing crop embeddings
        self.vit_wrapper = ViTWrapper(device=DEFAULT_DEVICE)

    def update_trackers(self, frame_num: int, points: List[np.array], embeddings: List[np.array], **kwargs):
        """
        Update the tracker with new det_query and crops
        :param frame_num: the starting frame number of the batch
        :param points: points in the format [[x1,y1],[x2,y2],[[x1,y1],[x2,y2],[x3,y3...] in the normalized scale 0-1
        a collection of det_query detected in each last_updated_frame.
        :param embeddings: a numpy array of embeddings in the format [[emb1],[emb2],[emb3]...]
        :param kwargs:
        :return:
        """
        info(f"Updating tracks with {len(points)} points in frames batch starting at {frame_num}")

        labels = kwargs.get("labels", [])
        scores = kwargs.get("scores", [0. for _ in range(len(points))])
        boxes = kwargs.get("boxes", [])
        max_cost = kwargs.get("max_cost", 30)
        t_pts = np.zeros((len(self.open_trackers), 2))
        t_emb = np.zeros((len(self.open_trackers), ViTWrapper.VECTOR_DIMENSIONS))
        d_emb = embeddings

        # If there are no det_query, return
        if len(points) == 0:
            return

        open_tracks = [t for t in self.open_trackers if not t.is_closed(frame_num)]
        info(f"Updating {len(open_tracks)} open tracks")

        # Get predicted det_query and embeddings from existing trackers
        for i, t in enumerate(open_tracks):
            t_pts[i] = t.predict()
            t_emb[i] = t.embedding

        points = np.array(points)
        d_emb = np.array(d_emb)

        # Associate the new det_query with the existing tracks
        costs = associate(detection_pts=points, detection_emb=d_emb, tracker_pts=t_pts, tracker_emb=t_emb)

        for cost, point, emb, label, score, box in zip(costs, points, d_emb, labels, scores, boxes):
            match = np.argmin(cost, axis=0)
            best_cost = cost[match]
            if len(open_tracks) > 0:
                info(f"Match {match} all costs {cost} for point {point} num trackers {len(open_tracks)}")
                if cost[match] < max_cost:
                    if len(open_tracks) > 0 and match < len(open_tracks):
                        info(f"Found match {match} with cost {best_cost} for point {point}")
                        info(f"Found match to track id {open_tracks[match].track_id} cost {best_cost} for point {point}")
                        open_tracks[match].update(label, point, emb, frame_num, box=box, score=score)
                else:
                    info(f"Match too high {best_cost} > {max_cost}; creating new track {self.next_track_id} for point {point}")
                    self.open_trackers.append(Track(self.next_track_id, label, point, emb, frame_num, self.image_width, self.image_height, box=box, score=score, **kwargs))
                    self.next_track_id += 1
            else:
                self.open_trackers.append(Track(self.next_track_id, label, point, emb, frame_num, self.image_width, self.image_height, box=box, score=score, **kwargs))
                self.next_track_id += 1

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
        for d in detections:
            if not Path(d["crop_path"]).exists():
                continue

            image = cv2.imread(d["crop_path"])
            gray_crop = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Choose the best keypoint to track based on SIFT
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(gray_crop, None)
            keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)
            if len(keypoints) == 0:
                # Choose the center of the crop if no keypoints are found
                x = image.shape[1] // 2
                y = image.shape[0] // 2
            else:
                x, y = keypoints[0].pt

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
            scale = (bbox[2] - bbox[0]) / 224
            x = x * scale
            y = y * scale
            x += bbox[0]
            y += bbox[1]
            # Adjust to 0-1 scale
            x /= self.image_width
            y /= self.image_height
            frame_num = d["frame"]
            label = d["class_name"]
            score = d["score"]
            bbox = [bbox[0] / self.image_width, bbox[1] / self.image_height, bbox[2] / self.image_width, bbox[3] / self.image_height]

            # Add the best keypoint to the query
            info(f"Adding query for {x}, {y} {label} in frame {frame_num}")
            if frame_num not in det_query:
                det_query[frame_num] = []
                crop_query[frame_num] = []
            det_query[frame_num].append([x, y, label, score, bbox])
            crop_query[frame_num].append(d["crop_path"])

        return self._update_batch(frame_range, frames, det_query, crop_query, **kwargs)

    def _update_batch(self, frame_range: Tuple[int, int], frames: np.ndarray, det_query: Dict, crop_query: Dict, **kwargs):
        """
        Update the tracker with the new frames, detections and crops of the detections
        :param frame_range: a tuple of the starting and ending frame numbers
        :param frames: numpy array of frames in the format [frame1, frame2, frame3...]
        :param det_query: a dictionary of detections in the format {frame_num: [[x1,y1,label,score,bbox],[x2,y2,label,score,bbox],[[x1,y1,label,score,bbox]]...]} in the normalized scale 0-1
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
        for f, d in det_query.items():
            # TODO: replace with parallel processing
            info(f"Computing embeddings for frame {f} {crop_query[f]}")
            labels = [det[2] for det in d]
            q_emb[f] = compute_embedding_vits(self.vit_wrapper, crop_query[f], labels)
            for det in d:
                queries.append([f - frame_range[0], det[0], det[1]])

        if len(queries) == 0:
            return []

        # Put the queries and frames into tensors and run the model with the backward tracking option which is
        # more accurate than the forward/online only tracking
        # Convert the queries to the model scale
        queries = np.array(queries)
        queries[:, 1] *= self.model_width
        queries[:, 2] *= self.model_height
        queries_t = torch.tensor(queries, dtype=torch.float32)
        frames = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2)
        if DEFAULT_DEVICE == "cuda":
            queries_t = queries_t.cuda()
            frames = frames.cuda()  # / 255.0

        video_chunk = frames.unsqueeze(0)  # Shape: (B, T, C, H, W)
        info(f"Running co-tracker model with {len(queries)} queries frames {frame_range}")
        pred_pts, pred_visibilities = self.offline_model(video_chunk, queries=queries_t[None], backward_tracking=True)
        pred_pts, pred_visibilities = pred_pts.cpu().numpy(), pred_visibilities.cpu().numpy()
        # Convert the queries  back to the normalized scale
        pred_pts[:, :, :, 0] /= self.model_width
        pred_pts[:, :, :, 1] /= self.model_height

        # Update with predictions
        for f in range(frame_range[0], frame_range[1], 1):
            if len(pred_pts) == 0:
                continue
            pts = pred_pts[:, f - frame_range[0], :, :]
            if len(pts) == 0:
                continue

            # Filter out the det_query that are not visible
            pred_visibilities_in_frame = pred_visibilities[:, f - frame_range[0], :]
            filtered_pts = pts[pred_visibilities_in_frame]

            # Create empty embeddings for the predicted det_query since this is just pt tracking
            empty_emb = np.zeros((len(filtered_pts), ViTWrapper.VECTOR_DIMENSIONS))
            self.update_trackers(f, filtered_pts, empty_emb, **kwargs)

        # Update with the queries - these seed new tracks and update existing tracks
        for f in range(frame_range[0], frame_range[1]):
            if f not in det_query:
                continue
            labels_in_frame = [d[2] for d in det_query[f]]
            scores_in_frame = [d[3] for d in det_query[f]]
            boxes_in_frame = [d[4] for d in det_query[f]]
            queries_in_frame = [d[:2] for d in det_query[f]]
            debug(f"Updating with queries {queries_in_frame} in frame {f}")
            self.update_trackers(f, queries_in_frame, q_emb[f], labels=labels_in_frame, scores=scores_in_frame, boxes=boxes_in_frame,**kwargs)

            # Close any tracks ready to close
            i = len(self.open_trackers)
            for t in reversed(self.open_trackers):
                i -= 1
                if t.is_closed(f):
                    info(f"Closing track {t.track_id}")
                    self.closed_trackers.append(t)
                    self.open_trackers.pop(i)

        return self.open_trackers + self.closed_trackers

    def purge_closed_tracks(self, frame_num: int):
        """
        Purge the closed tracks that are no longer needed
        :param frame_num: the current frame number
        :return:
        """
        # Close any tracks ready to close
        i = len(self.closed_trackers)
        for t in reversed(self.closed_trackers):
            i -= 1
            if t.is_closed(frame_num):
                info(f"Removing closed track {t.track_id}")
                self.closed_trackers.pop(i)


    def get_tracks(self):
        """
        Get the open and closed tracks
        :return: a list of open and closed tracks
        """
        return self.open_trackers + self.closed_trackers