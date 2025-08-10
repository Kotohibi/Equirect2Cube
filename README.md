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

## How it works
#### Original equirectangular image
<img width="1152" height="576" alt="frame_0001" src="https://github.com/user-attachments/assets/8f078f66-8eca-49a4-ad8f-f070b8c03063" />

#### Output cube images
|Position|File name|Image|
|----|----|----|
|left|[filename]_negx.png|<img width="288" height="288" alt="frame_0001_negx" src="https://github.com/user-attachments/assets/470262ed-76ee-442f-8b30-ac5997ade6ee" />|
|right|[filename]_posx.png|<img width="288" height="288" alt="frame_0001_posx" src="https://github.com/user-attachments/assets/d9c27bc3-a19c-4502-8023-d0d1a1a087db" />|
|bottom|[filename]_negy.png|<img width="288" height="288" alt="frame_0001_negy" src="https://github.com/user-attachments/assets/fef3bce3-ecf4-4456-8913-2177b7690923" />|
|top|[filename]_posy.png|<img width="288" height="288" alt="frame_0001_posy" src="https://github.com/user-attachments/assets/1e207e07-1800-4789-9f20-f32334cc0a49" />|
|forward|[filename]_posz.png|<img width="288" height="288" alt="frame_0001_posz" src="https://github.com/user-attachments/assets/27211589-9b0f-4b60-af5f-71916b600e10" />|
|back|[filename]_negz.png|<img width="288" height="288" alt="frame_0001_negz" src="https://github.com/user-attachments/assets/0e822f3a-1211-419b-b98a-256c798b8ba6" />|


