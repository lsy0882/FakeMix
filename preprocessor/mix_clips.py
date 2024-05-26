from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, concatenate_audioclips
from pathlib import Path
import random
import os
import json

# Set a seed for reproducibility
random.seed(42)

# Set the base directory for the dataset and output directory
dataset_base_dir = Path("/home/lsy/laboratory/Research/FakeMix/data/FakeAVCeleb_preprocessed")
output_base_dir = Path("/home/lsy/laboratory/Research/FakeMix/data/FakeAVCeleb_mixed")

# Set the categories
original = 'RealVideo-RealAudio'
faked_categories = ['FakeVideo-FakeAudio', 'FakeVideo-RealAudio', 'RealVideo-FakeAudio']

# Function to mix clips from a specific faked category for each file_id and version
def mix_clips(file_id, faked_category, faked_version):
    # Initialize lists to hold the final set of video and audio clips, and the category info
    final_video_clips = []
    final_audio_clips = []
    category_info = []

    # Determine the number of clips based on the number of files in the faked version directory
    faked_clips_dir = dataset_base_dir / faked_category / 'clips' / file_id / faked_version
    num_clips = len(os.listdir(faked_clips_dir))

    # Assuming there is only one version in the original category
    original_versions = os.listdir(dataset_base_dir / original / 'clips' / file_id)
    original_version = original_versions[0] if original_versions else None

    # For each clip index, select a clip from either the original or the faked category
    use_fake_for_remaining = False  # Flag to indicate if we should use fake clips for remaining indices

    for clip_index in range(num_clips):
        random_flag = random.choice([True, False])
        if not use_fake_for_remaining and random_flag:
            # Try to use the original clip
            video_clip_path = dataset_base_dir / original / 'clips' / file_id / original_version / f"{clip_index:03d}.mp4"
            audio_clip_path = dataset_base_dir / original / 'audio' / file_id / original_version / f"{clip_index:03d}.wav"
            chosen_category = original
            # If the original clip doesn't exist, use fake for this and all subsequent clips
            if not video_clip_path.exists():
                use_fake_for_remaining = True
        if use_fake_for_remaining or not random_flag:
            # Use the faked clip
            video_clip_path = faked_clips_dir / f"{clip_index:03d}.mp4"
            audio_clip_path = dataset_base_dir / faked_category / 'audio' / file_id / faked_version / f"{clip_index:03d}.wav"
            chosen_category = faked_category

        # Load the video clip and the corresponding audio clip
        video_clip = VideoFileClip(str(video_clip_path))
        audio_clip = AudioFileClip(str(audio_clip_path))

        # Append the clips to the lists
        final_video_clips.append(video_clip)
        final_audio_clips.append(audio_clip)
        
        # Append the chosen category to the category_info list
        category_info.append({
            'second': clip_index,
            'frames': [int(clip_index * video_clip.fps), int((clip_index + 1) * video_clip.fps - 1)],
            'category': chosen_category
        })

    # Concatenate all video clips and audio clips
    if final_video_clips:
        final_video = concatenate_videoclips(final_video_clips, method="compose")
        final_audio = concatenate_audioclips(final_audio_clips)
        
        # You need to explicitly set fps for the final video if it's not already set
        if not hasattr(final_video, 'fps') or final_video.fps is None:
            # You can set this to the fps of the original clips, or another value if appropriate
            final_video.fps = video_clip.fps  # 'video_clip.fps' is from the original video

        # Create the output directory if it doesn't exist
        output_dir = output_base_dir / faked_category / file_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write the final video to the file
        output_video_path = output_dir / f"{faked_version}_mixed.mp4"
        final_video.write_videofile(str(output_video_path), codec='libx264', audio_codec='aac')
        
        output_audio_path = output_dir / f"{faked_version}_mixed.wav"
        final_audio.write_audiofile(str(output_audio_path), codec="pcm_s16le")

        # Save the category info as a JSON file
        info_output_path = output_dir / f"{faked_version}_mixed.json"
        with open(info_output_path, 'w') as f:
            json.dump(category_info, f)

        # Close all clips to release resources
        final_video.close()
        
    for clip in final_video_clips:
        clip.close()
    for audio_clip in final_audio_clips:
        audio_clip.close()

# Iterate over each file_id and version in the faked categories and mix clips
for faked_category in faked_categories:
    for file_id_folder in (dataset_base_dir / faked_category / 'clips').iterdir():
        file_id = file_id_folder.name
        for version_folder in (dataset_base_dir / faked_category / 'clips' / file_id).iterdir():
            faked_version = version_folder.name
            mix_clips(file_id, faked_category, faked_version)
