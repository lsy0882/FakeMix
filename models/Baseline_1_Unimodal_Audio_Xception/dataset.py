import os
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
        self.category_to_idx_audio = {
            "FakeVideo-FakeAudio": 0,
            "FakeVideo-RealAudio": 1,
            "RealVideo-FakeAudio": 0,
            "RealVideo-RealAudio": 1
        }
        
        # Load train dataset
        if self.data_partition == "train":
            self.data_path = os.path.join(self.data_path, self.data_partition)
            for category in tqdm(os.listdir(self.data_path), desc='Loading categories'):
                category_path = os.path.join(self.data_path, category)
                type_audio_path = os.path.join(category_path, 'audio')

                for person_id in tqdm(os.listdir(type_audio_path), desc=f'Processing clips', leave=False):
                    person_audio_path = os.path.join(type_audio_path, person_id)

                    for file_id in os.listdir(person_audio_path):
                        file_audio_path = os.path.join(person_audio_path, file_id)
                        audio_files = sorted([f for f in os.listdir(file_audio_path) if f.endswith('.wav')])

                        for i in range(len(audio_files)):
                            clip_audio_path = os.path.join(file_audio_path, audio_files[i])
                            
                            self.samples.append((
                                clip_audio_path, 
                                self.category_to_idx_audio[category]
                            ))

        # Load test dataset
        elif self.data_partition == "test":
            self.data_path = os.path.join(self.data_path, self.data_partition)
            for category in tqdm(os.listdir(self.data_path), desc='Loading categories'):
                category_path = os.path.join(self.data_path, category)
                for person_id in tqdm(os.listdir(category_path), desc=f'Processing clips', leave=False):
                    person_id_path = os.path.join(category_path, person_id)
                    full_audio_paths = sorted([f for f in os.listdir(person_id_path) if f.endswith('.wav')])
                    annotation_paths = sorted([f for f in os.listdir(person_id_path) if f.endswith('.json')])
                    for idx in range(len(full_audio_paths)):
                        full_audio_path = os.path.join(person_id_path, full_audio_paths[idx])
                        annotation_path = os.path.join(person_id_path, annotation_paths[idx])
                        
                        with open(annotation_path, 'r') as ann_file:
                            annotations = json.load(ann_file)
                        
                        self.samples.append((
                            full_audio_path, 
                            self.category_to_idx_audio[category], 
                            annotations
                        ))


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get item train dataset
        if self.data_partition == "train":
            clip_audio_path, audio_class = self.samples[idx]
            audio, sr = librosa.load(clip_audio_path, sr=None)
            mfcc_features = self.process_audio(audio, sr)
            return {
                'mfcc_data': mfcc_features,
                'mfcc_class': torch.tensor(audio_class, dtype=torch.long)
            }

        # Get item test dataset
        elif self.data_partition == "test":
            full_audio_path, audio_class, annotations = self.samples[idx]
            audio, sr = librosa.load(full_audio_path, sr=None)

            mfcc_segments_data = []
            mfcc_segments_class = []
            for ann in annotations:
                clip_start_frame, clip_end_frame = ann['frames']
                fps = clip_end_frame - clip_start_frame + 1
                
                # TODO: modify annotation (0 sec -> [0, 1] sec) & calculate time
                audio_start = clip_start_frame / fps
                audio_end = (clip_end_frame + 1) / fps
                audio_segment = audio[int(audio_start * sr):int(audio_end * sr)]
                mfcc_segment_tensor = self.process_audio(audio_segment, sr)
                mfcc_class = self.category_to_idx_audio[ann['category']]
                
                mfcc_segments_data.append(mfcc_segment_tensor)
                mfcc_segments_class.append(torch.tensor(mfcc_class))
            
            return {
                'mfcc_data': torch.stack(mfcc_segments_data),
                'mfcc_class': torch.stack(mfcc_segments_class),
                'mfcc_path': full_audio_path
            }
    
    def process_audio(self, audio_data, sr):
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13) # ndarray [n_mfcc, T]
        mfcc = Image.fromarray(mfcc, mode='L') # [n_mfcc, T, 1]
        mfcc = Image.merge("RGB", (mfcc, mfcc, mfcc)) # [n_mfcc, T, 3]
        mfcc = self.preprocessing_func['audio'](mfcc) # torch [3, n_mfcc=224, T=224]
        return mfcc