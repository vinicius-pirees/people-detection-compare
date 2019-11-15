# people-detection-compare
Project to compare different object (people) detection techniques


#### Calculating mAP
```
python calculate_mean_ap.py \
    --ground-truth-boxes ~/personal-git/people-detection-compare/resources/virat_dataset/VIRAT_S_010000_00_000000_000165_ground_truth_boxes.json \
    --predicted-boxes ~/personal-git/people-detection-compare/results/VIRAT_S_010000_00_000000_000165_ssd_predicted_boxes.json
```