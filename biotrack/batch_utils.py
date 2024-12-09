# biotrack, CC-BY-NC license
# Filename: biotrack/batch_utils.py
# Description:  Miscellaneous functions mostly for example purposes

from pathlib import Path

import numpy as np


def find_files(directory: Path, extensions: list):
    files = list(directory.rglob("*"))
    valid_files = [f for f in files if f.suffix in extensions]
    return valid_files


def media_to_stack(media_path: Path, frame_count=None, resize=None):
    import imageio
    import numpy as np
    import cv2

    frames = []

    if not media_path.exists():
        print(f"{media_path} does not exist")
        return None, 0

    if media_path.is_dir():  # Assumes images
        valid_extensions = [".jpg", ".jpeg", ".png"]
        files = list(media_path.rglob("*"))
        valid_files = [f for f in files if f.suffix in valid_extensions]
        if len(valid_files) == 0:
            print(f"No valid files found in {media_path}")
            return None, 0

        for f in sorted(valid_files):
            print(f"Processing {f}")
            im = cv2.imread(str(f))
            if resize:
                im = cv2.resize(im, resize)
            frames.append(np.array(im))
    else:  # Assumes video file
        try:
            reader = imageio.get_reader(media_path.as_posix())
        except Exception as e:
            print(f"Error opening video file: {e}")
            return None, 0

        for i, im in enumerate(reader):
            if frame_count is not None and i >= frame_count:
                break
            if resize:
                im = cv2.resize(im, resize)
            frames.append(np.array(im))

    num_frames = len(frames)
    video_chunk = np.array(frames)

    return video_chunk, num_frames


if __name__ == "__main__":
    from pathlib import Path

    # Get the path to the parent directory of this file
    parent_dir = Path(__file__).parent.parent.parent / "data" / "video"
    media_path = parent_dir / "V4318_20201208T203419Z_h264_tl_3.mp4"

    video, num_frames = media_to_stack(media_path, frame_count=3, resize=(640, 360))


def percent_coverage(mask1, mask2):
    """ Calculate percent coverage between two binary masks."""
    if mask1.shape != mask2.shape:
        raise ValueError("Mask dimensions must match")

    # Ensure masks are binary
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    # Calculate overlap and union
    overlap = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    # Handle the edge case of empty masks
    if union == 0:
        return 100.0 if overlap == 0 else 0.0

    return np.round((overlap / union) * 100)


def reshape_transform(tensor):
    result = tensor[:, 1:, :].reshape(tensor.size(0), 14, 14, tensor.size(2))
    result = result.permute(0, 3, 1, 2)  # Transpose to channels-first
    return result
