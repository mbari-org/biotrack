# Tracker based on BioClip and CoTracker3 

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

- [BioClip](https://imageomics.github.io/bioclip/)
- [CoTracker3](https://cotracker3.github.io/)