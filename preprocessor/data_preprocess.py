import os
import dlib
import cv2
import pandas as pd
import numpy as np
from moviepy.editor import VideoFileClip, ImageSequenceClip

error_log_path = 'preprocess_error_log.xlsx'

# Call face detector
detector = dlib.get_frontal_face_detector()

def append_error_log(new_errors):
    if os.path.exists(error_log_path):
        # Read the existing data and append the new errors
        existing_errors = pd.read_excel(error_log_path)
        combined_errors = pd.concat([existing_errors, new_errors], ignore_index=True)
    else:
        combined_errors = new_errors

    # Write the combined errors back to the Excel file
    combined_errors.to_excel(error_log_path, index=False)

def split_video(video_path, output_dir, category, identity, video_id, crop_face_option, vid_ext='.mp4', aud_ext='.wav'):
    # Initialize subclip variable
    subclip = None
    
    # Create a VideoFileClip object
    clip = VideoFileClip(video_path)
    
    # Calculate the number of clips && fps && size
    number_of_clips = int(clip.duration)
    clip_fps = int(clip.fps)
    clip_width, clip_height = clip.size
    
    # Create directories for clips and audio if they don't exist
    clips_dir = os.path.join(output_dir, category, 'clips', identity, video_id)
    audio_dir = os.path.join(output_dir, category, 'audio', identity, video_id)
    os.makedirs(clips_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    
    # Check if clips directory already contains files, if so, skip processing
    if os.listdir(clips_dir):
        print(f"Clips already extracted for video {video_id}, skipping...")
        clip.close()  # Close the main video file as we're skipping processing
        return

    # Iterate through the number of clips and extract 1-second clips
    for i in range(number_of_clips):
        try:
            # Define the start and end of the clip
            start_time = i
            end_time = i + 1

            # Extract the subclip
            subclip = clip.subclip(start_time, end_time)
            
            # Define the output path for the clip and audio
            clip_output_path = os.path.join(clips_dir, f"{i:03d}{vid_ext}")
            audio_output_path = os.path.join(audio_dir, f"{i:03d}{aud_ext}")
            
            if crop_face_option:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(clip_output_path, fourcc, 25, (224, 224))
                face_frames = []
                selected_frames = np.linspace(0, clip.duration, 25, endpoint=False)  # Select 25 time points evenly spaced

                for t in selected_frames:
                    frame = subclip.get_frame(t)
                    face_rects, scores, idx = detector.run(frame, 0)
                    if face_rects:
                        rect = face_rects[0]  # Take the first detected face
                        x1, y1, x2, y2 = max(0, rect.left()), max(0, rect.top()), min(clip_width, rect.right()), min(clip_height, rect.bottom())
                        crop_img = frame[y1:y2, x1:x2]
                        crop_img = cv2.resize(crop_img, (224, 224), interpolation=cv2.INTER_AREA)
                        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
                    else:
                        if face_frames:
                            crop_img = face_frames[-1]  # Reuse the last face frame if no new face is detected
                        else:
                            resize_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                            crop_img = cv2.cvtColor(resize_frame, cv2.COLOR_RGB2BGR)
                    
                    face_frames.append(crop_img)  # Append the face frame or the reused frame to the list

                # Check if we have exactly 25 frames, throw an error if not
                if len(face_frames) != 25:
                    raise ValueError(f"Expected 25 frames, but got {len(face_frames)} frames for video {video_id}")

                for frame in face_frames:  # Write the frames to the video
                    video_writer.write(frame)

                video_writer.release()
                audio_clip = subclip.audio.set_start(start_time).set_end(end_time)
                audio_clip.write_audiofile(audio_output_path, codec="pcm_s16le")
            else:
                # Write the video clip
                subclip.write_videofile(clip_output_path, codec="libx264", audio_codec="aac", remove_temp=True)
                
                # Extract the audio from the subclip and write it to a file
                subclip.audio.write_audiofile(audio_output_path, codec="pcm_s16le")

        except Exception as e:
            # Log the error for the current clip
            error_data = {
                'video_path': video_path,
                'clip_index': range(i, number_of_clips),
                'error_message': str(e)
            }
            # Create a DataFrame for the new error
            new_error_df = pd.DataFrame([error_data])
            # Append the new error to the log
            append_error_log(new_error_df)
        
    # Close the main clip to free up resources
    clip.close()

def preprocess_fakeavceleb(root_dir, data_dir, output_dir, crop_face_option):
    # Loop through the directory structure
    for category in os.listdir(data_dir):
        category_path = os.path.join(data_dir, category)
        if os.path.isdir(category_path) and category == "RealVideo-FakeAudio":
            for race in os.listdir(category_path):
                race_path = os.path.join(category_path, race)
                if os.path.isdir(race_path):
                    for gender in os.listdir(race_path):
                        gender_path = os.path.join(race_path, gender)
                        if os.path.isdir(gender_path):
                            for identity in os.listdir(gender_path):
                                identity_path = os.path.join(gender_path, identity)
                                if os.path.isdir(identity_path):
                                    for filename in os.listdir(identity_path):
                                        if filename.endswith('.mp4'):
                                            video_path = os.path.join(identity_path, filename)
                                            video_id = filename.split('.')[0]
                                            split_video(video_path, output_dir, category, identity, video_id, crop_face_option)

# Example usage (the paths need to be adjusted to your actual data locations)
root_directory = '/home/lsy/laboratory/Research/idea4_MDFD'
data_directory = '/home/lsy/laboratory/Research/idea4_MDFD/data/FakeAVCeleb'  # Replace with the path to the root of the dataset
output_directory = '/home/lsy/laboratory/Research/idea4_MDFD/data/FakeAVCeleb_preprocessed_onlyface_fixlen'           # Replace with your desired output path

crop_face = True
# Run the preprocessing function
preprocess_fakeavceleb(root_directory, data_directory, output_directory, crop_face)

# The above function call is commented out because this environment does not have access to the dataset files.
# You can uncomment and run this function in your local environment where the dataset is accessible.