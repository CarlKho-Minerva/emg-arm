#!/bin/sh
# Stitch one recorded run into an mp4 at 15 frames/s.   ./stitch.sh data/follow/<run>
D=$(cd "$1" && pwd); ls "$D"/*.jpg | sort | sed "s/^/file '/; s/$/'/" > /tmp/stitch_list.txt
ffmpeg -loglevel error -y -r 15 -f concat -safe 0 -i /tmp/stitch_list.txt -vf scale=960:-2 -pix_fmt yuv420p "$D/run.mp4" && echo "$D/run.mp4"
