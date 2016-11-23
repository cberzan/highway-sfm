#!/usr/bin/env bash

# Clean dataset dir.
# (Script copied from the opensfm repo and modified.)
# Usage: `./clean.sh opensfm-dataset-dir`

trash=$1/trash/`date -u +"%Y-%m-%dT%H:%M:%SZ"`
mkdir -p $trash

mv -vf $1/reconstruction*.json $trash
mv -vf $1/exif $trash
mv -vf $1/matches $trash
mv -vf $1/sift $trash
mv -vf $1/surf $trash
mv -vf $1/akaze* $trash
mv -vf $1/root* $trash
mv -vf $1/hahog $trash
mv -vf $1/camera_models.json $trash
mv -vf $1/reference_lla.json $trash
mv -vf $1/profile.log $trash
mv -vf $1/navigation_graph.json $trash
mv -vf $1/plot_inliers $trash
mv -vf $1/depthmaps $trash
mv -vf $1/tracks.csv $trash
