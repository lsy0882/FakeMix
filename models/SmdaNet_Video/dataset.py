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
        self.category_to_idx_video = {
            "FakeVideo-FakeAudio": 0,
            "FakeVideo-RealAudio": 0,
            "RealVideo-FakeAudio": 1,
            "RealVideo-RealAudio": 1
        }
        
        # Load train dataset
        if self.data_partition == "train":
            self.data_path = os.path.join(self.data_path, self.data_partition)
            for category in tqdm(os.listdir(self.data_path), desc='Loading categories'):
                category_path = os.path.join(self.data_path, category)
                clip_video_path = os.path.join(category_path, 'clips')

                for person_id in tqdm(os.listdir(clip_video_path), desc=f'Processing clips', leave=False):
                    person_clips_path = os.path.join(clip_video_path, person_id)

                    for file_id in os.listdir(person_clips_path):
                        file_clips_path = os.path.join(person_clips_path, file_id)
                        clips_files = sorted([f for f in os.listdir(file_clips_path) if f.endswith('.mp4')])

                        for i in range(len(clips_files) - 1):  # except last index
                            current_clip_path = os.path.join(file_clips_path, clips_files[i])
                            next_clip_path = os.path.join(file_clips_path, clips_files[i + 1])
                            
                            self.samples.append((
                                (current_clip_path, next_clip_path),
                                self.category_to_idx_video[category],
                            ))
                        
        # Load test dataset
        elif self.data_partition == "test":
            self.data_path = os.path.join(self.data_path, self.data_partition)
            for category in tqdm(os.listdir(self.data_path), desc='Loading categories'):
                category_path = os.path.join(self.data_path, category)
                for person_id in tqdm(os.listdir(category_path), desc=f'Processing clips', leave=False):
                    person_id_path = os.path.join(category_path, person_id)
                    clip_paths = sorted([f for f in os.listdir(person_id_path) if f.endswith('.mp4')])
                    annotation_paths = sorted([f for f in os.listdir(person_id_path) if f.endswith('.json')])
                    for idx in range(len(clip_paths)):
                        full_video_path = os.path.join(person_id_path, clip_paths[idx])
                        annotation_path = os.path.join(person_id_path, annotation_paths[idx])
                        
                        with open(annotation_path, 'r') as ann_file:
                            annotations = json.load(ann_file)
                        
                        self.samples.append((
                            full_video_path, 
                            self.category_to_idx_video[category], 
                            annotations
                        ))


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get item train dataset
        if self.data_partition == "train":
            (clip_video_path_1, clip_video_path_2), video_class= self.samples[idx]
            video_frames_1, _ = self.process_video(clip_video_path_1)
            video_frames_2, _ = self.process_video(clip_video_path_2)
            return {
                'video_data': torch.stack([video_frames_1, video_frames_2], dim=0),
                'video_class': torch.tensor(video_class, dtype=torch.long)
            }

        # Get item test dataset
        elif self.data_partition == "test":
            full_video_path, video_class, annotations = self.samples[idx]
            video_tensor, fps = self.process_video(full_video_path)

            video_segments_data = []
            video_segments_class = []
            for ann in annotations:
                clip_start_frame, clip_end_frame = ann['frames']
                start_idx = min(clip_start_frame, len(video_tensor) - 1)
                end_idx = min(clip_end_frame + 1, len(video_tensor))
                clip_data_tensor = video_tensor[start_idx:end_idx]
                clip_class = self.category_to_idx_video[ann['category']]
                
                video_segments_data.append(clip_data_tensor)
                video_segments_class.append(torch.tensor(clip_class))

            return {
                'video_data': torch.stack(video_segments_data),
                'video_class': torch.stack(video_segments_class),
                'video_path': full_video_path
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
