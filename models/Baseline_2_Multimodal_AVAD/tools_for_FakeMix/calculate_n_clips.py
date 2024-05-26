import cv2
import os

input_txt_path = '/home/lsy/laboratory/Research/FakeMix/Baseline_2_Multimodal_AVAD/tools_for_FakeMix/FakeMix_mp4_paths.txt'
output_txt_path = '/home/lsy/laboratory/Research/FakeMix/Baseline_2_Multimodal_AVAD/tools_for_FakeMix/FakeMix_n_clips.txt'

with open(input_txt_path, 'r') as file:
    video_paths = file.readlines()

total_clips = 0
with open(output_txt_path, 'w') as outfile:
    for video_path in video_paths:
        video_path = video_path.strip()
        if not os.path.exists(video_path):
            print(f"File not found: {video_path}")
            continue

        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        clips_count = frame_count // 25
        total_clips += clips_count

        outfile.write(f"{video_path}: {clips_count} clips\n")

    outfile.write(f"Total clips: {total_clips}\n")

print("Processing complete. Results saved to:", output_txt_path)