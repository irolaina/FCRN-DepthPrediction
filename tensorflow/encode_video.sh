#!/bin/sh

ffmpeg -r 30 -f image2 -s 304x288 -i output/fcrn_cv/frame%06d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p  output/frame.mp4
ffmpeg -r 30 -f image2 -s 304x288 -i output/fcrn_cv/pred%06d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p  output/pred.mp4
ffmpeg -i output/frame.mp4 -i output/pred.mp4 -filter_complex "hstack,format=yuv420p" -c:v libx264 -crf 25 output/output.mp4
echo "[encoding] Removing Temporary Files..."
rm output/frame.mp4
rm output/pred.mp4
echo "[encoding] Done."