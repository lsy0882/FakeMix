import shutil
from random import sample
from pathlib import Path

def copy_random_samples(data_dir, output_dir, num_samples_per_category):
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all categories
    categories = [p for p in data_dir.iterdir() if p.is_dir()]

    # Copy random samples from each category
    for category in categories:
        # Paths for clips and audio within the category
        clips_dir = category / 'clips'
        audio_dir = category / 'audio'

        # Get all unique file_ids from the clips directory
        file_ids = {p.parent.name for p in clips_dir.rglob('*.mp4')}

        # Convert the set to a list and sample random file_ids
        sampled_file_ids = sample(list(file_ids), min(num_samples_per_category, len(file_ids)))

        # Copy the directories for each sampled file_id
        for file_id in sampled_file_ids:
            # Source directories
            src_clips_dir = clips_dir / file_id
            src_audio_dir = audio_dir / file_id

            # Destination directories
            dest_clips_dir = output_dir / category.name / 'clips' / file_id
            dest_audio_dir = output_dir / category.name / 'audio' / file_id

            # Copy the clip and audio directories
            shutil.copytree(src_clips_dir, dest_clips_dir, dirs_exist_ok=True)
            shutil.copytree(src_audio_dir, dest_audio_dir, dirs_exist_ok=True)
# Example usage
num_samples = 5  # Number of random samples per category to copy
data_directory = Path("/home/lsy/laboratory/Research/idea4_MDFD/data/FakeAVCeleb_mixed_onlyface")  # Adjust to the path to your data
output_directory = Path("/home/lsy/laboratory/Research/idea4_MDFD/temp/test")  # Adjust to your desired output path
copy_random_samples(data_directory, output_directory, num_samples)