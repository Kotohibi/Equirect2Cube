# Convert equirectangular to cube images

## Preparation
- If you want to convert videos in equirectangular, you need to extract frame images from the video.
#### Example for using FFmpeg to extract frames
- ffmpeg -i input.mp4 -vf fps=1 frame_%04d.png

## Execution
#### Example
- python .\e_to_c.py --input-dir .\input\ --out-dir .\out\ --workers 8 --overlap 40 --face-size 3840

#### Arguments
|Name|Description|
|----|----|
|--input-dir|the folder containing frames|
|--out-dir|output folder to store cube images|
|--workers|multi-process number. less than CPU cores|
|--overlap|30-40% is better for SfM|
|--face-size|resolution for each image|
