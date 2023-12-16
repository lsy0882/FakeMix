import os
import cv2
import librosa
import torch
import json
from tqdm import tqdm
from PIL import Image

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, data_partition, preprocessing_func):
        self.data_path = data_path
        self.data_partition = data_partition
        self.preprocessing_func = preprocessing_func
        self.samples = []
        self.category_to_idx_video_audio = {
            "FakeVideo-FakeAudio": 0,
            "FakeVideo-RealAudio": 1,
            "RealVideo-FakeAudio": 2,
            "RealVideo-RealAudio": 3
        }
        
        # Load train dataset
        if self.data_partition == "train":
            self.data_path = os.path.join(self.data_path, self.data_partition)
            for category in tqdm(os.listdir(self.data_path), desc='Loading categories'):
                category_path = os.path.join(self.data_path, category)
                clip_video_path = os.path.join(category_path, 'clips')
                clip_audio_path = os.path.join(category_path, 'audio')

                for person_id in tqdm(os.listdir(clip_video_path), desc=f'Processing clips', leave=False):
                    person_clips_path = os.path.join(clip_video_path, person_id)
                    person_audio_path = os.path.join(clip_audio_path, person_id)

                    for file_id in os.listdir(person_clips_path):
                        file_clips_path = os.path.join(person_clips_path, file_id)
                        file_audio_path = os.path.join(person_audio_path, file_id)
                        clips_files = sorted([f for f in os.listdir(file_clips_path) if f.endswith('.mp4')])
                        audio_files = sorted([f for f in os.listdir(file_audio_path) if f.endswith('.wav')])

                        # Make sure we have the same number of audio and video files
                        min_length = min(len(clips_files), len(audio_files))

                        for i in range(min_length - 1):  # except last index
                            current_clip_path = os.path.join(file_clips_path, clips_files[i])
                            next_clip_path = os.path.join(file_clips_path, clips_files[i + 1])
                            current_audio_path = os.path.join(file_audio_path, audio_files[i])
                            next_audio_path = os.path.join(file_audio_path, audio_files[i + 1])
                            
                            self.samples.append((
                                (current_clip_path, next_clip_path),
                                (current_audio_path, next_audio_path),
                                self.category_to_idx_video_audio[category]
                            ))
                        
        # Load test dataset
        elif self.data_partition == "test":
            self.data_path = os.path.join(self.data_path, self.data_partition)
            for category in tqdm(os.listdir(self.data_path), desc='Loading categories'):
                category_path = os.path.join(self.data_path, category)
                for person_id in tqdm(os.listdir(category_path), desc=f'Processing clips', leave=False):
                    person_id_path = os.path.join(category_path, person_id)
                    clip_paths = sorted([f for f in os.listdir(person_id_path) if f.endswith('.mp4')])
                    audio_paths = sorted([f for f in os.listdir(person_id_path) if f.endswith('.wav')])
                    annotation_paths = sorted([f for f in os.listdir(person_id_path) if f.endswith('.json')])
                    for idx in range(len(clip_paths)):
                        full_video_path = os.path.join(person_id_path, clip_paths[idx])
                        full_audio_path = os.path.join(person_id_path, audio_paths[idx])
                        annotation_path = os.path.join(person_id_path, annotation_paths[idx])
                        
                        with open(annotation_path, 'r') as ann_file:
                            annotations = json.load(ann_file)
                        
                        self.samples.append((
                            full_video_path, 
                            full_audio_path, 
                            self.category_to_idx_video_audio[category], 
                            annotations
                        ))


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get item train dataset
        if self.data_partition == "train":
            (clip_video_path_1, clip_video_path_2), (clip_audio_path_1, clip_audio_path_2), video_audio_class = self.samples[idx]
            video_frames_1, _ = self.process_video(clip_video_path_1)
            video_frames_2, _ = self.process_video(clip_video_path_2)
            audio_1, sr_1 = librosa.load(clip_audio_path_1, sr=None)
            mfcc_features_1 = self.process_audio(audio_1, sr_1)
            audio_2, sr_2 = librosa.load(clip_audio_path_2, sr=None)
            mfcc_features_2 = self.process_audio(audio_2, sr_2)
            return {
                'video_data': torch.stack([video_frames_1, video_frames_2], dim=0),
                'mfcc_data': torch.stack([mfcc_features_1, mfcc_features_2], dim=0),
                'video_mfcc_class': torch.tensor(video_audio_class, dtype=torch.long)
            }

        # Get item test dataset
        elif self.data_partition == "test":
            full_video_path, full_audio_path, video_audio_class, annotations = self.samples[idx]
            video_tensor, fps = self.process_video(full_video_path)
            audio, sr = librosa.load(full_audio_path, sr=None)

            video_segments_data = []
            mfcc_segments_data = []
            video_mfcc_segments_class = []
            for ann in annotations:
                clip_start_frame, clip_end_frame = ann['frames']
                start_idx = min(clip_start_frame, len(video_tensor) - 1)
                end_idx = min(clip_end_frame + 1, len(video_tensor))
                clip_data_tensor = video_tensor[start_idx:end_idx]
                
                audio_start = clip_start_frame / fps
                audio_end = (clip_end_frame + 1) / fps
                audio_segment = audio[int(audio_start * sr):int(audio_end * sr)]
                mfcc_segment_tensor = self.process_audio(audio_segment, sr)
                
                clip_mfcc_class = self.category_to_idx_video_audio[ann['category']]
                
                video_segments_data.append(clip_data_tensor)
                mfcc_segments_data.append(mfcc_segment_tensor)
                video_mfcc_segments_class.append(torch.tensor(clip_mfcc_class))
            
            return {
                'video_data': torch.stack(video_segments_data),
                'video_path': full_video_path,
                'mfcc_data': torch.stack(mfcc_segments_data),
                'mfcc_path': full_audio_path,
                'video_mfcc_class': torch.stack(video_mfcc_segments_class),
            }
            
    def process_video(self, video_path):
        video_clip = cv2.VideoCapture(video_path)
        fps = video_clip.get(cv2.CAP_PROP_FPS)
        frames = []
        while True:
            ret, frame = video_clip.read() # ndarray [H, W, C=3]
            if not ret:
                break
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) # [H, W, C=3]
            frame = self.preprocessing_func['video'](pil_image) # torch [C=3, H=224, W=224]
            frames.append(frame)
        video_clip.release()
        frames = torch.stack(frames)
        return frames, fps # torch [T=25, C=3, H=224, W=224]
    
    def process_audio(self, audio_data, sr):
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13) # ndarray [n_mfcc, T]
        mfcc = Image.fromarray(mfcc, mode='L') # [n_mfcc, T, 1]
        mfcc = Image.merge("RGB", (mfcc, mfcc, mfcc)) # [n_mfcc, T, 3]
        mfcc = self.preprocessing_func['audio'](mfcc) # torch [3, n_mfcc=224, T=224]
        return mfcc