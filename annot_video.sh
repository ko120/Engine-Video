#!/bin/bash

# python annotate_video.py \
#   --input /content/drive/MyDrive/AI video analytic/Data/Easy/124441_10-13min.mp4\
#   --annot sam3_124441_10_13.json \
#   --output 124441_annotated.mp4 \
#   --json-out 124441_annotated.json \
#   --horizon 30 \
#   --lookback 5

python visualize_traj_predictions.py \
  --video "/content/drive/MyDrive/AI video analytic/Data/Easy/124441_10-13min.mp4" \
  --xml /content/engineVideo/predictions.xml \
  --output 124441_traj_annotated.mp4