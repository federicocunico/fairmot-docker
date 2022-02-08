ffmpeg -framerate 20 -i 'frames/%d.jpg' -c:v libx264 -pix_fmt yuv420p output_frames.mp4
ffmpeg -framerate 20 -i 'bview/%d.jpg' -c:v libx264 -pix_fmt yuv420p output_bview.mp4
