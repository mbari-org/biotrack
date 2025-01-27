from biotrack.tracker import BioTracker
from biotrack.batch_utils import media_to_stack

if __name__ == "__main__":
    import json
    import cv2
    from pathlib import Path

    parent_dir = Path(__file__).parent
    media_path = parent_dir / "data" / "video" / "V4318_20201208T203419Z_h264_tl_3.mp4"
    out_video_path = parent_dir / "data" / "video" / "V4318_20201208T203419Z_h264_tl_3_biotrack.mp4"
    detections_path = parent_dir / "data" / "detections"
    crops_path = parent_dir / "data" / "images" / "crops"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Load the video and convert it to a stack of frames
    frame_stack, num_frames = media_to_stack(media_path, frame_count=60, resize=(640, 360))
    frame_stack_full, _ = media_to_stack(media_path, frame_count=60)
    out_video = cv2.VideoWriter(out_video_path, fourcc, 1, (1920, 1080))

    # Initialize the tracker
    tracker = BioTracker(1920, 1080)
    window_len = 60  #Max number of frames to track at once; this is the window length

    tracks = None

    # Run the tracker for all the frames in windows of window_len
    for i in range(0, num_frames, window_len):
        frames = frame_stack[i : i + window_len]
        frame_full = frame_stack_full[i : i + window_len]

        # Get all the detections in the window to pass to the tracker - these are called the queries
        detections = []
        for j in range(len(frames)):
            frame_num = i + j
            # Load the det_query for the last_updated_frame
            detections_file = detections_path / f"{frame_num}.json"
            if not detections_file.exists():
                print(f"No det_query for frame {frame_num}")
                continue

            data = json.loads(detections_file.read_text())
            for loc in data:
                if j >= num_frames:
                    continue

                loc["crop_path"] = (crops_path / loc["crop_path"]).as_posix()
                loc["frame"] = frame_num
                loc["score"] = loc["confidence"]
                detections.append(loc)

        i_e = min(i + window_len - 1, num_frames)  # handle the end of the video
        print(f"Tracking frames {i} to {i_e}")
        tracks = tracker.update_batch((i, i_e), frames, detections=detections, max_frames=60, max_empty_frames=3, max_cost=0.3)

        # Display the tracks for the window
        for j in range(len(frames)):
            frame_num = i + j
            frame = frame_full[j]
            # Convert the frame to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            print(f"Displaying last_updated_frame {frame_num}")
            for track in tracks:
                pt, label, box, score = track.get(frame_num, rescale=False)
                if pt is not None:
                    print(f"Drawing point {pt},{label} for track {track.id}")
                    pt = (int(pt[0] * 1920), int(pt[1] * 1080))
                    center = (int(pt[0]) + 10, int(pt[1]))
                    radius = 10
                    color = (255, 255, 255)
                    thickness = 1
                    frame = cv2.circle(frame, center, radius, color, thickness)
                    # Draw the track track_id with the label, e.g. 1:Unknown
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    frame = cv2.putText(frame, f"{track.id}:{label}{score:.2f}", center, font, fontScale, color, thickness, cv2.LINE_AA)

                if box is not None:
                    # Draw the box
                    frame = cv2.rectangle(frame,
                                          (int(box[0] * 1920), int(box[1] * 1080)),
                                          (int(box[2] * 1920), int(box[3] * 1080)),
                                          (0, 255, 0), 1)

            if frame is not None:
                cv2.imshow("Frame", frame)
                cv2.waitKey(250)
                out_video.write(frame)

        # Print basic info on closed tracks, and purge them. Purging is not necessary for the tracker to work,
        # but it is good practice for memory management
        closed_tracks = [track for track in tracks if track.is_closed()]
        if len(closed_tracks) > 0:
            for track in closed_tracks:
                print(f"Closed track {track.id} at frame {i_e}")
                best_frame, best_pt, best_label, best_box, best_score = track.get_best()
                print(f"Best track {track.id} is {best_pt},{best_box},{best_label},{best_score} in frame {best_frame}")
            tracker.purge_closed_tracks()

    out_video.release()
    # Print out any remaining open tracks
    tracks = tracker.get_tracks()
    if len(tracks) > 0:
        for track in tracks:
            print(f"Track {track.id} at frame {i_e}")
            best_frame, best_pt, best_label, best_box, best_score = track.get_best()
            print(f"Best track {track.id} is {best_pt},{best_box},{best_label} in frame {best_frame}")