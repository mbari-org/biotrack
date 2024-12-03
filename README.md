# Tracker based on CoTracker3 

This is a tracker based on the CoTracker3 algorithm with some modifications for Biodiversity lab video.
This is work in progress.  This uses two transformer-based models to track objects in a video; one model for appearance and 
one for tracking points motion. The CoTracker3 model is run in an offline mode for best performance.


# Installation

```bash
poetry install
```

# Run the tracker on a sample video

```bash
export PYTHONPATH=. && poetry run python examples/video.py
```

Should see a window with the video and the tracked points, something similar to this:

![video](https://github.com/mbari-org/biotrack/blob/main/examples/data/video/V4318_20201208T203419Z_h264_tl_3_biotrack.gif)


# References
 
- [CoTracker3](https://cotracker3.github.io/)