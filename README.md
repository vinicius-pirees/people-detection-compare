# people-detection-compare
Project to compare different object (people) detection techniques


#### Run the detectors

##### Normal
```bash
python mobile_ssd.py --only-moving-frames y
```

##### With Tiles
```bash
python mobile_ssd.py \
    --detect-on-tiles y \
    --tiles-x 2 \
    --tiles-y 3 \
    --margin-percent 0.25 \
    --only-moving-frames y
```
    

#### Calculating mAP
```bash
python calculate_mean_ap.py \
    --ground-truth-boxes ~/personal-git/people-detection-compare/resources/virat_dataset/VIRAT_S_010000_00_000000_000165_ground_truth_boxes.json \
    --predicted-boxes ~/personal-git/people-detection-compare/results/VIRAT_S_010000_00_000000_000165_mobile_ssd_predicted_boxes.json
```

#### Display Resulting Bounding Boxes
```bash
python display_result.py \
    --video-file ~/personal-git/people-detection-compare/resources/virat_dataset/VIRAT_S_010000_00_000000_000165.mp4 \
    --ground-truth-boxes ~/personal-git/people-detection-compare/resources/virat_dataset/VIRAT_S_010000_00_000000_000165_ground_truth_boxes.json \
    --predicted-boxes ~/personal-git/people-detection-compare/results/VIRAT_S_010000_00_000000_000165_mobile_ssd_predicted_boxes.json
```


make -j5
./fpdw ./inria_detector.xml ~/Downloads/proposal.jpeg