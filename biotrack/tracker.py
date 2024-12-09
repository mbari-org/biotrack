# biotrack, CC-BY-NC license
# Filename: biotrack/tracker/tracker.py
# Description:  Main tracker class for tracking objects with points, label and embeddings using the CoTracker model and BioClip ViT embeddings
from dbm.dumb import error

from PIL import Image

import torch
import piexif
from typing import List, Dict, Tuple
import numpy as np
from cotracker.predictor import CoTrackerPredictor

from biotrack.assoc import associate_track_pts_emb, associate_trace_pts
from biotrack.embedding import ViTWrapper
from biotrack.track import Track
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

    def update_trackers_queries(self, frame_num: int, keypoints: np.array, labels: List[str], scores: np.array, boxes: np.array,  d_emb: np.array, **kwargs):
        """
        Update the tracker with the queries. This will seed new tracks if they cannot be associated with existing tracks
        :param boxes:  the bounding boxes of the queries in the format [[x1,y1,x2,y2],[x1,y1,x2,y2]...] in the normalized scale 0-1
        :param scores:  the scores of the queries in the format [score1, score2, score3...] in the normalized scale 0-1
        :param labels:  the labels of the queries in the format [label1, label2, label3...]
        :param frame_num:  the starting frame number of the batch
        :param keypoints:  points in the format [[[x1,y1],[x2,y2],[[x1,y1],[x2,y2],[x3,y3...], [x1,y1],[x2,y2],[[x1,y1],[x2,y2],[x3,y3...], one per track in the normalized scale 0-1
        :param kwargs:
        :return:
        """
        if len(keypoints) == 0:
            return

        info(f"Query updating {len(keypoints)} tracks with query points in frame {frame_num}")
        max_cost = kwargs.get("max_cost", 0.4)

        if len(self.open_trackers) == 0:
            new_tracks = []
            # Create new tracks the length of the keypoints
            while len(new_tracks) < len(keypoints):
                new_tracks.append(Track(self.next_track_id, self.image_width, self.image_height, **kwargs))
                self.next_track_id += 1

            # Get the newly created tracks and initialize
            for j, data in enumerate(zip(new_tracks, keypoints, d_emb, boxes, labels, scores)):
                new_tracks[j], points, emb, box, label, score = data
                for i, pt in enumerate(points[0]):
                    new_tracks[j].init(i, label, pt, emb, frame_num, box=box, score=score)

            self.open_trackers.extend(new_tracks)
            return

        t_pts, t_emb = self.get_tracks_detail()

        # Flatten the keypoints and associate the new points with the existing track traces
        keypoints = np.array([item for sublist in keypoints for item in sublist[0]])
        # assignment, costs = associate_trace_pts(detection_pts=keypoints, trace_pts=t_pts)
        assignment, costs = associate_track_pts_emb(detection_pts=keypoints, detection_emb=d_emb, trace_pts=t_pts, tracker_emb=t_emb)
        if len(assignment) == 0:
            return

        open_traces = [t.get_traces() for t in self.open_trackers]
        open_traces = [item for sublist in open_traces for item in sublist]
        unassigned = []
        for t_idx, d_idx in assignment:
            cost = costs[t_idx, d_idx]
            if t_idx < len(open_traces) and cost < max_cost:
                open_traces[t_idx].update_pt(keypoints[d_idx], frame_num)
            else:
                unassigned.append(d_idx)

        # Create new tracks for the unassigned points.  There are NUM_KP points per track and unassigned should be multiples of NUM_KP
        if len(unassigned) % Track.NUM_KP != 0:
            error(f"Unassigned points {len(unassigned)} is not a multiple of {Track.NUM_KP}")
            return

        for i in range(0, len(unassigned), Track.NUM_KP):
            info(f"Creating new track {self.next_track_id} at {frame_num}")
            track = Track(self.next_track_id, self.image_width, self.image_height, **kwargs)
            unassigned_idx = unassigned[i:i + Track.NUM_KP]
            for j, d_idx in enumerate(unassigned_idx):
                box = boxes[d_idx // Track.NUM_KP]
                label = labels[d_idx // Track.NUM_KP]
                score = scores[d_idx // Track.NUM_KP]
                track.init(j, label, keypoints[d_idx], d_emb[d_idx // Track.NUM_KP], frame_num, box=box, score=score)
            self.open_trackers.append(track)
            self.next_track_id += 1


    def get_tracks_detail(self):
        # Get predicted track traces and embeddings from existing trackers
        t_pts = np.zeros((len(self.open_trackers)*Track.NUM_KP, 2))
        t_emb = np.zeros((len(self.open_trackers)*Track.NUM_KP, self.vit_wrapper.vector_dimensions))
        j = 0
        for i, t in enumerate(self.open_trackers):
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

        max_cost = kwargs.get("max_cost", 0.1)

        # If there are no tracked points
        if len(points) == 0:
            return

        t_pts, _ = self.get_tracks_detail()

        debug(f"Associating {len(points)} points with {len(t_pts)} traces")

        if len(t_pts) == 0: # no open traces to associate with
            return

        # Associate the new points with the existing track traces
        open_traces = [t.get_traces() for t in self.open_trackers][0]
        assignment, costs = associate_trace_pts(detection_pts=points, trace_pts=t_pts)

        for t_idx, d_idx in assignment:
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
        def correct_keypoints(top_kps, crop_paths):
            correct_kpts = top_kps
            for i, data in enumerate(zip(top_kps, crop_paths)):
                top_kp, crop_path = data
                with Image.open(crop_path) as img:
                    exif_dict = img._getexif()
                    if exif_dict is not None:
                        user_comment = exif_dict[piexif.ExifIFD.UserComment]
                        user_comment = user_comment.decode("utf-8")
                        if user_comment.startswith("bbox:"):
                            bbox_str = user_comment.split("bbox:")[1]
                            bbox = eval(bbox_str)
                            # Actual size of the crop which is rescaled to 224x224 - this is used to find
                            # the top left x, y in the original image for the query
                            scale_x = (bbox[2] - bbox[0]) / 224
                            scale_y = (bbox[3] - bbox[1]) / 224
                            for j, pt in enumerate(top_kp[0]):
                                x, y = pt
                                x *= scale_x
                                y *= scale_y
                                x += bbox[0]
                                y += bbox[1]
                                # Adjust to 0-1 scale
                                x /= self.image_width
                                y /= self.image_height
                                correct_kpts[i][0][j] = [x, y]

            return correct_kpts

        unique_frames = np.unique([d["frame"] for d in detections])

        det_query = {f: [] for f in unique_frames}
        image_query = {f: [] for f in unique_frames}

        for i in unique_frames:
            boxes = [[d['x'],d['y'],d['xx'],d['xy']] for d in detections if d["frame"] == i]
            images = [d['crop_path'] for d in detections if d["frame"] == i]
            embeddings, predicted_classes, predicted_scores, keypoints = self.vit_wrapper.process_images(images)
            # Remove any data that has no keypoints
            # Get the index of the keypoints that are empty
            empty_idx = [i for i, kpts in enumerate(keypoints) if len(kpts) == 0]
            if len(empty_idx) > 0:
                info(f"Removing empty keypoints {empty_idx}")
            embeddings = [emb for i, emb in enumerate(embeddings) if i not in empty_idx]
            predicted_classes = [p for i, p in enumerate(predicted_classes) if i not in empty_idx]
            predicted_scores = [p for i, p in enumerate(predicted_scores) if i not in empty_idx]
            if len(embeddings) == 0: # No data found
                info(f"No valid keypoints found for frame {i}")
                det_query.pop(i)
                image_query.pop(i)
                continue
            correct_kpts = correct_keypoints(keypoints, images)
            predicted_classes = [p[0] for p in predicted_classes]
            predicted_scores = [p[0] for p in predicted_scores]
            info(f"Adding query for {correct_kpts} in frame idx {i}")
            det_query[i].append([correct_kpts, predicted_classes, predicted_scores, boxes, embeddings])
            image_query[i].append(images)

        return self._update_batch(frame_range, frames, det_query, image_query, **kwargs)

    def _update_batch(self, frame_range: Tuple[int, int], frames: np.ndarray, det_query: Dict, crop_query: Dict, save:bool = False, **kwargs):
        """
        Update the tracker with the new frames, detections and crops of the detections
        :param frame_range: a tuple of the starting and ending frame numbers
        :param frames: numpy array of frames in the format [frame1, frame2, frame3...]
        :param det_query: a dictionary of detection keypoints, labels and boxes in the format {frame_num: [[[x1,y1],[x2,y2],...label,score,bbox],[[x1,y1],[x2,y2],label,score,bbox]...]} in the normalized scale 0-1
        :param crop_query: a dictionary of crop paths in the format {frame_num: [crop_path1, crop_path2, crop_path3...]}
        :param save: save the tracks to video
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
        frame_numbers = det_query.keys()
        for f in frame_numbers:
            queries_in_frame = [d[0] for d in det_query[f]][0]
            embeddings_in_frame = [d[4] for d in det_query[f]]
            for kpts in queries_in_frame[0]:
                for kpt in kpts:
                    queries.append([int(f - frame_range[0]), kpt[0], kpt[1]])
            q_emb[f] = embeddings_in_frame[0]
        debug(f"Queries: {queries}")

        if len(queries) == 0:
            info("No queries found")
            return []

        # Update with the queries - these seed new tracks and update existing tracks
        # Because the model predicts point backward, points that are not yet associated with
        # a box can be visible, so here we create a mask of when the points start to be visible for each frame
        query_vis = np.ndarray((len(frames), len(queries)), dtype=bool)
        query_vis.fill(False)
        vis_idx = 0
        for i, f in enumerate(frame_numbers):
            queries_in_frame = [d[0] for d in det_query[f]]
            labels_in_frame = [d[1] for d in det_query[f]] # Get the top label and score
            scores_in_frame = [d[2] for d in det_query[f]]
            boxes_in_frame = [d[3] for d in det_query[f]]
            debug(f"Updating with queries {queries_in_frame} in frame {f}")
            self.update_trackers_queries(f, queries_in_frame[0], labels_in_frame[0], scores_in_frame[0], boxes_in_frame[0], q_emb[f], **kwargs)
            self.check(f)
            # Set the points that may be visible in all the frames from the current frame to the end of the frame range
            for ii in range(f, frame_range[1]):
                if ii < f:
                    continue
                for j in range(0, len(labels_in_frame[0]*Track.NUM_KP)):
                    query_vis[ii][vis_idx + j] = True
            vis_idx += len(labels_in_frame[0]*Track.NUM_KP)

        # Put the queries and frames into tensors and run the model with the backward tracking option which is
        # more accurate than the forward/online only tracking
        queries_ = np.array(queries)
        Q = queries_.copy()
        Q[:, 1] = queries_[:, 1] * self.model_width
        Q[:, 2] = queries_[:, 2] * self.model_height

        queries_t = torch.tensor(Q, dtype=torch.float32)
        frames = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2)

        # Put the data on the cuda device
        queries_t = queries_t.to(self.device)
        frames = frames.to(self.device)

        video_chunk = frames.unsqueeze(0)  # Shape: (B, T, C, H, W)
        info(f"Running co-tracker model with {len(queries)} queries frames {frame_range}")
        pred_tracks, pred_vis = self.offline_model(video_chunk, queries=queries_t[None], backward_tracking=True)

        # Get the final visibility mask by combining the query visibility and the model visibility
        query_vis = torch.tensor(query_vis, dtype=torch.bool).to(self.device)
        query_vis = query_vis.unsqueeze(0).expand(pred_vis.shape[0], -1, -1)
        pred_vis = pred_vis * query_vis

        if save:
            from cotracker.utils.visualizer import Visualizer
            vis = Visualizer(
                save_dir='./',
                linewidth=1,
                fps=1,
                show_first_frame=0,
                mode='rainbow',
                tracks_leave_trace=10,
            )
            vis.visualize(
                video=video_chunk,
                tracks=pred_tracks,
                visibility=query_vis,
                filename=f'traces{frame_range[0]}-{frame_range[1]}')

        pred_tracks, pred_vis = pred_tracks.cpu().numpy(), pred_vis.cpu().numpy()

        # Convert the predictions back to the normalized scale
        pred_tracks[:, :, :, 0] /= self.model_width
        pred_tracks[:, :, :, 1] /= self.model_height

        # Update with predictions
        for f in frame_numbers:
            tracks = pred_tracks[:, f - frame_range[0], :, :]
            vis = pred_vis[:, f - frame_range[0], :]
            vis_tracks = tracks[vis]
            if len(vis_tracks) == 0:
                continue
            debug(f"Found {len(vis_tracks)} tracks that are visible in frame {f}")
            self.update_trackers_pred(f + frame_range[0], vis_tracks, **kwargs)
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