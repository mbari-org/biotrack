# Tracker based on CoTracker3 

This is a tracker based on the CoTracker3 algorithm with some modifications to make it work with the Biodiversity lab video.
This is work in progress.

# Installation

```bash
poetry install
```

# Run the tracker on a sample video

```bash
export PYTHONPATH=. && poetry run python examples/video.py
```

Should see a window with the video and the tracked points, something simiar to this:

![video](https://github.com/mbari-org/biotrack/blob/main/examples/data/video/V4318_20201208T203419Z_h264_tl_3_biotrack.gif)


# References
 
- [CoTracker3](https://cotracker3.github.io/)