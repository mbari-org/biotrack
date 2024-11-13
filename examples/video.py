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
    out_video = cv2.VideoWriter(out_video_path, fourcc, 1, (1920, 1080))

    # Initialize the tracker
    tracker = BioTracker(1920, 1080)
    window_len = 8

    tracks = None

    # Run the tracker for all the frames in windows of window_len
    for i in range(0, num_frames, window_len):
        frames = frame_stack[i : i + window_len]

        # Get all the detections in the window
        detections = []
        for j in range(len(frames)):
            frame_num = i + j
            # Load the detections for the last_updated_frame
            detections_file = detections_path / f"{frame_num}.json"
            if not detections_file.exists():
                print(f"No detections for frame {frame_num}")
                continue

            data = json.loads(detections_file.read_text())
            for loc in data:
                if j >= num_frames:
                    continue

                # Convert the x, y to the image coordinates and adjust crop path to the full path
                loc["x"] = loc["x"] * loc["image_width"]
                loc["y"] = loc["y"] * loc["image_height"]
                loc["xx"] = loc["xx"] * loc["image_width"]
                loc["xy"] = loc["xy"] * loc["image_height"]
                loc["crop_path"] = (crops_path / loc["crop_path"]).as_posix()
                loc["frame"] = frame_num
                detections.append(loc)

        i_e = min(i + window_len - 1, num_frames - 1)  # handle the end of the video
        print(f"Tracking frames {i} to {i_e}")
        tracks = tracker.update_batch((i, i_e), frames, detections=detections)

        # Display the tracks for the window
        for j in range(len(frames)):
            frame_num = i + j
            print(f"Displaying last_updated_frame {frame_num}")
            image = parent_dir / "data" / "images" / f"{frame_num}.jpg"
            frame = cv2.imread(image.as_posix())
            for track in tracks:
                pt, label = track.get_point_label(frame_num)
                if pt is not None:
                    print(f"Drawing point {pt},{label}")
                    center = (int(pt[0]), int(pt[1]))
                    radius = 10
                    color = (255, 255, 255)
                    thickness = 1
                    frame = cv2.circle(frame, center, radius, color, thickness)
                    # Draw the track id with the label, e.g. 1:Unknown
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    frame = cv2.putText(frame, f"{track.id}:{label}", center, font, fontScale, color, thickness, cv2.LINE_AA)

            if frame is not None:
                cv2.imshow("Frame", frame)
                cv2.waitKey(250)
                out_video.write(frame)
