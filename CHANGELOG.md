# CHANGELOG


## v0.7.7 (2024-12-20)

### Performance Improvements

- Compute embedding in same block as model output to avoid forward compute twice
  ([`86f0bd9`](https://github.com/mbari-org/biotrack/commit/86f0bd90e67772db1e664a63581f57a8eb7c7a9a))


## v0.7.6 (2024-12-17)

### Build System

- Added missing transformer dep
  ([`4300da1`](https://github.com/mbari-org/biotrack/commit/4300da1287ad55b25b37e69cf6d32fffbecce808))

### Chores

- Change device id to gpu id kwargs
  ([`f743694`](https://github.com/mbari-org/biotrack/commit/f743694d072d7be46d3767e03c4a119adb169049))

- Renamed model arg to vits_model
  ([`0aa0c4b`](https://github.com/mbari-org/biotrack/commit/0aa0c4ba7ced52ae1a89d16f9195d19024d6488a))

### Performance Improvements

- Reduce to top 5 gcam for spped-up and slight mods of args for use in aipipeline
  ([`09e9e42`](https://github.com/mbari-org/biotrack/commit/09e9e4222e6bf773f88a0aab601d938aa25a5354))


## v0.7.5 (2024-12-12)


## v0.7.4 (2024-12-12)

### Bug Fixes

- Correct handling of different size cost association arrays
  ([`dd61fdd`](https://github.com/mbari-org/biotrack/commit/dd61fddb269845cfd4be1cc6ac0c7b1434b0d371))

- Correct trace to label assignment
  ([`3e654d8`](https://github.com/mbari-org/biotrack/commit/3e654d88501f0c710912b7414339c8860d45dccd))


## v0.7.3 (2024-12-12)

### Build System

- Better build for cuda deploy
  ([`47f93ca`](https://github.com/mbari-org/biotrack/commit/47f93ca79fa388819767eccb7aa031ac6ccf1782))

### Chores

- Minor change in example for clarity
  ([`f66145e`](https://github.com/mbari-org/biotrack/commit/f66145e6100bdbe767144dbd3b8916827c4d823e))

### Performance Improvements

- Improved keypoint assignment, remove IOU tracks, and save the second prediction from the gcam but
  keep the model prediction
  ([`6b586c3`](https://github.com/mbari-org/biotrack/commit/6b586c330e3b14fac5959a3fe1c612ead1cdf63a))


## v0.7.2 (2024-12-11)

### Bug Fixes

- Default to keypoint only (no embedding) cost and fix query visibility matrix
  ([`5eab5fc`](https://github.com/mbari-org/biotrack/commit/5eab5fc0af63dca520c0901ebee3d290af030b4e))


## v0.7.1 (2024-12-10)

### Bug Fixes

- Removed unused import
  ([`42cfd73`](https://github.com/mbari-org/biotrack/commit/42cfd733c68a00a7df6e645a01ef73d50f4f8b56))

### Build System

- Updated poetry
  ([`3bfcb34`](https://github.com/mbari-org/biotrack/commit/3bfcb343154a45475403efc187fcf9338827652b))


## v0.7.0 (2024-12-10)

### Features

- Added support for model config through model_name kwargs to BioTracker constructor and some other
  minor refactoring for clarity
  ([`0473a76`](https://github.com/mbari-org/biotrack/commit/0473a7689bae3f7555d7b8de12222d162feda9c8))


## v0.6.1 (2024-12-10)

### Bug Fixes

- Swap the score and coverage to display/save the score not coverage
  ([`06b9a5d`](https://github.com/mbari-org/biotrack/commit/06b9a5d3512d9f48c3189964131ee35d345f04b6))


## v0.6.0 (2024-12-09)

### Chores

- Updated license to CC-BY-NC license
  ([`e01bb2d`](https://github.com/mbari-org/biotrack/commit/e01bb2d51dad037f2bea249f71c46a7c7d8e0864))

### Documentation

- Updated readme
  ([`fedc966`](https://github.com/mbari-org/biotrack/commit/fedc96664a7f379f78908a98308cc31be6ff2ed5))

### Features

- Improved tracking with gradcam coverage activation map and Hungarian cost assignment
  ([`8cf6ed6`](https://github.com/mbari-org/biotrack/commit/8cf6ed6fd029cef9e26f953ac2fb438cb294ee41))


## v0.5.6 (2024-12-03)


## v0.5.5 (2024-11-22)

### Performance Improvements

- Added support for multiple keypoints, better CUDA support and top-3 prediction output
  ([`80c1aa4`](https://github.com/mbari-org/biotrack/commit/80c1aa4dbe35d731308cd64b847670bbd2883459))

- Weighted score for best label/score
  ([`4875e18`](https://github.com/mbari-org/biotrack/commit/4875e1848c8a380b1d2de68fb896f57019213003))


## v0.5.4 (2024-11-22)

### Performance Improvements

- Reduce the impact of the early detections by only considering the last 10 frames and update
  highest scoring label
  ([`ab14fce`](https://github.com/mbari-org/biotrack/commit/ab14fce8e6bd7ebf2082854c977f62e5e7c75944))


## v0.5.3 (2024-11-22)

### Performance Improvements

- Improved track closure and change debugging level
  ([`be86070`](https://github.com/mbari-org/biotrack/commit/be860706c88866e8d0eefa367483d8730fdc7570))


## v0.5.2 (2024-11-16)

### Performance Improvements

- Majority vote for best label and average score
  ([`4f3d5f3`](https://github.com/mbari-org/biotrack/commit/4f3d5f325176eec7404d5638d31df0b9860512e5))


## v0.5.1 (2024-11-16)

### Bug Fixes

- Correct filter of query points
  ([`751daa2`](https://github.com/mbari-org/biotrack/commit/751daa22889c806fca680d189f9fa2542370a773))

### Performance Improvements

- Omit zero value score from best label and move best frame to -3
  ([`5e75293`](https://github.com/mbari-org/biotrack/commit/5e7529360767917b70bd63f2102a64bd8cc0e298))


## v0.5.0 (2024-11-15)

### Features

- Add pass through of max_frames and max_empty_frames for track
  ([`321335d`](https://github.com/mbari-org/biotrack/commit/321335d3bb12562344b3b41890b4d7f848fc739a))


## v0.4.1 (2024-11-15)

### Documentation

- Updated gif with latest
  ([`c6a9c33`](https://github.com/mbari-org/biotrack/commit/c6a9c3305dd57c4d57c80f0fa50eba4c164bf749))

### Performance Improvements

- Change alpha weight to 0.5 and disable text embedding until have time to fully test
  ([`f8f4a61`](https://github.com/mbari-org/biotrack/commit/f8f4a616c8a39f93c25bd20a2dc6c088097dc29f))

### Refactoring

- Change to mostly normalized coordinates, except where needed
  ([`02bc124`](https://github.com/mbari-org/biotrack/commit/02bc12441cfa77ad5752ed2f45756437d3e5bdad))

- Move track clean to single point
  ([`e0cc692`](https://github.com/mbari-org/biotrack/commit/e0cc69273e180fef3bc87b539691783078ba1661))


## v0.4.0 (2024-11-14)


## v0.3.0 (2024-11-14)

### Features

- Rescale best
  ([`af64b0a`](https://github.com/mbari-org/biotrack/commit/af64b0a00553de180f322855ffc7b84b8e8908e8))


## v0.2.1 (2024-11-14)

### Features

- Return scores with other data for video overlay or output
  ([`379e719`](https://github.com/mbari-org/biotrack/commit/379e71901cfdd6a7f30f7314bd83c2fe05538262))


## v0.2.0 (2024-11-14)

### Performance Improvements

- Purge tracks, update with score and handle ending of video in example
  ([`c2a57ea`](https://github.com/mbari-org/biotrack/commit/c2a57ea02a3f8fb85f190dc146c1f8f3d7a9041c))


## v0.1.4 (2024-11-13)

### Bug Fixes

- Handle empty keypoints
  ([`a0e1aff`](https://github.com/mbari-org/biotrack/commit/a0e1aff948c6b8eb2710e720dd9dfaaac7449190))

### Features

- Added support for bounding box and online update of label
  ([`bbd2351`](https://github.com/mbari-org/biotrack/commit/bbd235174a176fce0e8647d9e2440efbca443c91))


## v0.1.3 (2024-11-13)

### Documentation

- Added missing gif
  ([`0973610`](https://github.com/mbari-org/biotrack/commit/0973610ddf8bcacf11c91a0b8c90dea2c2bfcda7))

### Performance Improvements

- Replace keypoint with SIFT, increase default cost to 30, and example defaults more realistic
  ([`ed4eb5f`](https://github.com/mbari-org/biotrack/commit/ed4eb5f200ee87349321704c259be9262464a951))


## v0.1.2 (2024-11-13)


## v0.1.1 (2024-11-13)

### Bug Fixes

- Order frames in media stack
  ([`a0691d5`](https://github.com/mbari-org/biotrack/commit/a0691d59b582cd9e3fedb28bbb9f99910c015f3c))

- Working GPU/CPU, improved tracking by reordering prediction before seed, and add blob detector for
  point selection
  ([`019e77b`](https://github.com/mbari-org/biotrack/commit/019e77b8bbe91fb0f4f9497fc65cee65026840d7))

### Chores

- Removed unused code
  ([`804533b`](https://github.com/mbari-org/biotrack/commit/804533bc61ad5575e76ecc78c357c24814bc75d7))


## v0.1.0 (2024-11-13)

### Features

- Initial commit
  ([`ce75eae`](https://github.com/mbari-org/biotrack/commit/ce75eae88860875bc69fed313c2781e176a9fe0c))
