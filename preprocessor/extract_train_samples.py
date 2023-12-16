import os
import shutil
from random import sample

def copy_random_samples(data_dir, output_dir, num_samples_per_category):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over each category in the data directory
    for category in os.listdir(data_dir):
        category_path = os.path.join(data_dir, category)
        if not os.path.isdir(category_path):
            continue  # Skip if it's not a directory

        # Get all video IDs within the category
        video_ids = []
        identity_dirs = os.path.join(category_path, 'clips')
        for identity in os.listdir(identity_dirs):
            identity_path = os.path.join(identity_dirs, identity)
            if os.path.isdir(identity_path):
                video_ids.extend([os.path.join(identity, video_id) for video_id in os.listdir(identity_path)])

        # Randomly sample video IDs from the list
        sampled_video_ids = sample(video_ids, min(num_samples_per_category, len(video_ids)))

        # Copy each sampled video ID directory for both clips and audio
        for video_id in sampled_video_ids:
            # Define source and destination paths for the clips and audio
            clip_src_dir = os.path.join(data_dir, category, 'clips', video_id)
            audio_src_dir = os.path.join(data_dir, category, 'audio', video_id)
            clip_dest_dir = os.path.join(output_dir, category, 'clips', video_id)
            audio_dest_dir = os.path.join(output_dir, category, 'audio', video_id)

            # Ensure destination directories exist
            os.makedirs(clip_dest_dir, exist_ok=True)
            os.makedirs(audio_dest_dir, exist_ok=True)

            # Copy the directories
            shutil.copytree(clip_src_dir, clip_dest_dir, dirs_exist_ok=True)
            shutil.copytree(audio_src_dir, audio_dest_dir, dirs_exist_ok=True)

num_samples = 5  # Number of random samples per category to copy
data_directory = '/home/lsy/laboratory/Research/idea4_MDFD/data/FakeAVCeleb_preprocessed_onlyface'  # Adjust to the path to your data
output_directory = '/home/lsy/laboratory/Research/idea4_MDFD/temp'  # Adjust to your desired output path
copy_random_samples(data_directory, output_directory, num_samples)