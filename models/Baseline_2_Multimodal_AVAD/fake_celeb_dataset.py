import os
import pickle
import threading

import numpy as np
from torch.utils import data
from tqdm import tqdm
from torchvision.transforms import InterpolationMode
from load_audio import load_wav
from load_video import load_mp4
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

import torch
import json
import librosa

class FakeAVceleb(data.Dataset):

    def __init__(self, video_path, resize, fps, sample_rate, vid_len, phase, train=True, number_sample=1, lrs2=False, need_shift=False, lrs3=False, kodf=False, real=True, lavdf=False, vox_korea=False, random_shift=False, fixed_shift=False, shift=0,robustness=False, test=False):
        super(FakeAVceleb, self).__init__()
        self.resize = resize
        self.fps = fps
        self.sample_rate = sample_rate
        self.vid_len = vid_len
        self.train = train
        self.lrs2 = lrs2
        self.lrs3 = lrs3
        self.real = real
        self.kodf = kodf
        self.lavdf = lavdf
        self.random_shift = random_shift
        self.fixed_shift = fixed_shift
        self.shift = shift
        self.vox_korea = vox_korea
        self.robustness = robustness
        self.test = test
        if self.lrs2:
            self.data_path = '/datab/chfeng/mvlrs_v1/pretrain'
        elif self.lrs3:
            self.data_path = '/datab/chfeng/lrs3'
        elif self.kodf:
            if self.real:
                self.data_path = '/datab/chfeng/syncnet_python/real_set/pycrop'
            else:
                self.data_path =  '/datab/chfeng/syncnet_python/fake_set/pycrop'
        elif self.lavdf:
            self.data_path = '/datab/chfeng/lavdf'
        elif self.vox_korea:
            self.data_path = '/datad/chfeng/vox_korea'
        elif self.robustness:
            self.data_path = '/datad/chfeng/DeeperForensics-1.0'
        elif self.test:
            self.data_path = ''
        else:
            self.data_path = '/datab/chfeng/av_sync'
        self.phase = phase
        self.all_vids = video_path
        self.number_sample = number_sample
        self.need_shift = need_shift
        

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.all_vids)

    def __getitem__(self, index):
        'Generates one sample of data'
        assert index < self.__len__()

        vid_path = self.all_vids[index]
        if self.lrs2:
            if self.train:
                vid_path = vid_path.split('\n')[0]
            else:
                vid_path = vid_path.split(' ')[0]
        elif self.lrs3:
            vid_path = vid_path.split('.')[0]
        elif self.kodf:
            vid_path = vid_path
        elif self.lavdf:
            vid_path = vid_path
        elif self.vox_korea:
            vid_path = vid_path.split('.')[0]
        elif self.robustness:
            vid_path = vid_path.split('.')[0]
        elif self.test:
            vid_path = vid_path.split('.')[0]
        else:
            vid_path = vid_path.split('.')[0]
            vid_path = vid_path.split(' ')
            if len(vid_path) == 1:
                vid_path = vid_path[0]
            elif len(vid_path) == 2:
                vid_path = vid_path[0] + vid_path[1]
            else:
                raise Exception('That is impossible')
        #vid_name, vid_ext = os.path.splitext(vid_path)
        if self.train:
            vid_name = vid_path
        else:
            vid_name = vid_path

        # -- load video
        if self.kodf:
            vid_path_orig = os.path.join(self.data_path, vid_name + '.avi')
            vid_path_25fps = os.path.join(self.data_path, vid_name + '.mp4')
        else:
            vid_path_orig = os.path.join(self.data_path, vid_name + '.mp4')
            vid_path_25fps = os.path.join(self.data_path, vid_name + '.mp4')
        # -- reencode video to 25 fps
        
        command = (
            "ffmpeg -threads 1 -loglevel error -y -i {} -an -r 25 {}".format(
                vid_path_orig, vid_path_25fps))
        from subprocess import call
        
        cmd = command.split(' ')
        #print('Resampling {} to 25 fps'.format(vid_path_orig))
        #call(cmd)

        video = self.__load_video__(vid_path_25fps, resize=self.resize)

        aud_path = os.path.join(self.data_path, vid_name + '.wav')
        if not os.path.exists(aud_path):  # -- extract wav from mp4
            command = (
                ("ffmpeg -threads 1 -loglevel error -y -i {} "
                    "-async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {}")
                .format(vid_path_orig, aud_path))
            from subprocess import call
            cmd = command.split(' ')
            call(cmd)
        
        audio = load_wav(aud_path).astype('float32')
        print(audio.shape)

        fps = self.fps  # TODO: get as param?
        aud_fact = int(np.round(self.sample_rate / fps))
        audio, video = self.trunkate_audio_and_video(video, audio, aud_fact)
        assert aud_fact * video.shape[0] == audio.shape[0]
        audio = np.array(audio)
        #video = video[0:30, :, :, :]
        #audio = audio[0:(30*aud_fact)]
        if self.need_shift:
            if self.random_shift:
                shift = np.random.randint(-15, 15, 1)
            elif self.fixed_shift:
                shift = self.shift
        else:
            shift = np.array([0])
        true_shift = shift
        audio_len = audio.shape[0]
        '''
        if shift == 0:
            audio = audio
        elif shift < 0:
            audio[:(audio_len - shift*aud_fact)] = audio[shift*aud_fact:]
            audio[(audio_len - shift*aud_fact):] = 0
        elif shift > 0:
            audio[shift*aud_fact:] = audio[:(audio_len - shift*aud_fact)]
            audio[:shift*aud_fact] = 0
        '''
        assert aud_fact * video.shape[0] == audio.shape[0]
        video = video.transpose([3, 0, 1, 2])  # t h w c -> c t h w
        if shift[0] == 0:
            audio = np.pad(audio, (15*aud_fact, 15*aud_fact), 'constant', constant_values=(0,0))
        elif shift[0] > 0:
            shift = np.abs(shift[0])
            audio = np.pad(audio, ((15 + shift)*aud_fact, (15 - shift)*aud_fact), 'constant', constant_values=(0,0))
        elif shift[0] < 0:
            shift = np.abs(shift[0])
            audio = np.pad(audio, ((15 - shift)*aud_fact, (15 + shift)*aud_fact), 'constant', constant_values=(0,0))
        #audio = np.expand_dims(audio,axis=0)
        #video = np.expand_dims(video,axis=0)
        
        out_dict = {
            'video': video,
            'audio': audio,
            'sample': vid_path,
            'shift':true_shift
        }

        return out_dict

    def __load_video__(self, vid_path, resize=None):

        frames = load_mp4(vid_path)

        if resize:
            import torchvision
            from PIL import Image
            ims = [Image.fromarray(frm) for frm in frames]
            ims = [
                torchvision.transforms.functional.resize(im,
                                                         [resize, resize], 
                                                         interpolation=InterpolationMode.BICUBIC)
                for im in ims
            ]
            frames = np.array([np.array(im) for im in ims])

        return frames.astype('float32')

    def trunkate_audio_and_video(self, video, aud_feats, aud_fact):

        aud_in_frames = aud_feats.shape[0] // aud_fact

        # make audio exactly devisible by video frames
        aud_cutoff = min(video.shape[0], int(aud_feats.shape[0] / aud_fact))

        aud_feats = aud_feats[:aud_cutoff * aud_fact]
        aud_in_frames = aud_feats.shape[0] // aud_fact

        min_len = min(aud_in_frames, video.shape[0])

        # --- trunkate all to min
        video = video[:min_len]
        aud_feats = aud_feats[:min_len * aud_fact]
        if not aud_feats.shape[0] // aud_fact == video.shape[0]:
            import ipdb
            ipdb.set_trace(context=20)

        return aud_feats, video


class FakeMix(data.Dataset):

    def __init__(self, video_path, resize, fps, sample_rate, vid_len, phase, train=True, number_sample=1, lrs2=False, need_shift=False, lrs3=False, kodf=False, real=True, lavdf=False, vox_korea=False, random_shift=False, fixed_shift=False, shift=0,robustness=False, test=False):
        super(FakeMix, self).__init__()
        self.resize = resize
        self.fps = fps
        self.sample_rate = sample_rate
        self.vid_len = vid_len
        self.train = train
        self.lrs2 = lrs2
        self.lrs3 = lrs3
        self.real = real
        self.kodf = kodf
        self.lavdf = lavdf
        self.random_shift = random_shift
        self.fixed_shift = fixed_shift
        self.shift = shift
        self.vox_korea = vox_korea
        self.robustness = robustness
        self.test = test
        if self.lrs2:
            self.data_path = '/datab/chfeng/mvlrs_v1/pretrain'
        elif self.lrs3:
            self.data_path = '/datab/chfeng/lrs3'
        elif self.kodf:
            if self.real:
                self.data_path = '/datab/chfeng/syncnet_python/real_set/pycrop'
            else:
                self.data_path =  '/datab/chfeng/syncnet_python/fake_set/pycrop'
        elif self.lavdf:
            self.data_path = '/datab/chfeng/lavdf'
        elif self.vox_korea:
            self.data_path = '/datad/chfeng/vox_korea'
        elif self.robustness:
            self.data_path = '/datad/chfeng/DeeperForensics-1.0'
        elif self.test:
            self.data_path = ''
        else:
            self.data_path = '/datab/chfeng/av_sync'
        self.phase = phase
        self.all_vids = video_path
        self.number_sample = number_sample
        self.need_shift = need_shift
        

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.all_vids)

    def __getitem__(self, index):
        'Generates one sample of data'
        assert index < self.__len__()

        vid_path = self.all_vids[index]
        if self.lrs2:
            if self.train:
                vid_path = vid_path.split('\n')[0]
            else:
                vid_path = vid_path.split(' ')[0]
        elif self.lrs3:
            vid_path = vid_path.split('.')[0]
        elif self.kodf:
            vid_path = vid_path
        elif self.lavdf:
            vid_path = vid_path
        elif self.vox_korea:
            vid_path = vid_path.split('.')[0]
        elif self.robustness:
            vid_path = vid_path.split('.')[0]
        elif self.test:
            vid_path = vid_path.split('.')[0]
        else:
            vid_path = vid_path.split('.')[0]
            vid_path = vid_path.split(' ')
            if len(vid_path) == 1:
                vid_path = vid_path[0]
            elif len(vid_path) == 2:
                vid_path = vid_path[0] + vid_path[1]
            else:
                raise Exception('That is impossible')
        #vid_name, vid_ext = os.path.splitext(vid_path)
        if self.train:
            vid_name = vid_path
        else:
            vid_name = vid_path

        # -- load video
        if self.kodf:
            vid_path_orig = os.path.join(self.data_path, vid_name + '.avi')
            vid_path_25fps = os.path.join(self.data_path, vid_name + '.mp4')
        else:
            vid_path_orig = os.path.join(self.data_path, vid_name + '.mp4')
            vid_path_25fps = os.path.join(self.data_path, vid_name + '.mp4')
        # -- reencode video to 25 fps
        
        command = (
            "ffmpeg -threads 1 -loglevel error -y -i {} -an -r 25 {}".format(
                vid_path_orig, vid_path_25fps))
        from subprocess import call
        
        cmd = command.split(' ')
        #print('Resampling {} to 25 fps'.format(vid_path_orig))
        #call(cmd)

        video = self.__load_video__(vid_path_25fps, resize=self.resize)
        fps = self.fps  # TODO: get as param?
        video_segments = []
        # 비디오를 초당 프레임 수(fps)에 따라 분할
        for i in range(0, len(video), fps):
            segment = video[i:i+fps]
            video_segments.append(segment)
        
        # # 결과 출력 (각 세그먼트의 길이 확인)
        # for idx, segment in enumerate(video_segments):
        #     print(f"[Video] Segment {idx+1}: {len(segment)} frames")

        aud_path = os.path.join(self.data_path, vid_name + '.wav')
        if not os.path.exists(aud_path):  # -- extract wav from mp4
            command = (
                ("ffmpeg -threads 1 -loglevel error -y -i {} "
                    "-async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {}")
                .format(vid_path_orig, aud_path))
            from subprocess import call
            cmd = command.split(' ')
            call(cmd)
        
        # audio = load_wav(aud_path).astype('float32')
        audio, sr = librosa.load(aud_path, sr=None, mono=False)
        audio = audio.mean(axis=0)  ### Multi-channel -> Single-channel
        samples_per_second = sr
        audio_segments = []

        # 전체 오디오를 1초 길이의 조각으로 나누기
        for start in range(0, len(audio), samples_per_second):
            end = start + samples_per_second
            # 배열 범위를 초과하지 않도록 조각
            segment = audio[start:end]
            audio_segments.append(segment)

        # # 결과 출력 (각 세그먼트의 길이 확인)
        # for i, segment in enumerate(audio_segments):
        #     print(f"[Audio] Segment {i+1}: {len(segment)} samples")

        
        # aud_fact = int(np.round(sr / fps))
        aud_fact = int(np.round(16000 / fps))
        out_dicts = []
        annotation = os.path.join(self.data_path, vid_name + '.json')
        with open(annotation, 'r') as file:
            annotation_data = json.load(file)
        categories = [item['category'] for item in annotation_data]
        
        for idx, (video, audio, category) in enumerate(zip(video_segments, audio_segments, categories)):
        
            audio, video = self.trunkate_audio_and_video(video, audio, aud_fact)
            assert aud_fact * video.shape[0] == audio.shape[0]
            audio = np.array(audio)
            #video = video[0:30, :, :, :]
            #audio = audio[0:(30*aud_fact)]
            if self.need_shift:
                if self.random_shift:
                    shift = np.random.randint(-15, 15, 1)
                elif self.fixed_shift:
                    shift = self.shift
            else:
                shift = np.array([0])
            true_shift = shift
            audio_len = audio.shape[0]
            '''
            if shift == 0:
                audio = audio
            elif shift < 0:
                audio[:(audio_len - shift*aud_fact)] = audio[shift*aud_fact:]
                audio[(audio_len - shift*aud_fact):] = 0
            elif shift > 0:
                audio[shift*aud_fact:] = audio[:(audio_len - shift*aud_fact)]
                audio[:shift*aud_fact] = 0
            '''
            assert aud_fact * video.shape[0] == audio.shape[0]
            video = video.transpose([3, 0, 1, 2])  # t h w c -> c t h w
            if shift[0] == 0:
                audio = np.pad(audio, (15*aud_fact, 15*aud_fact), 'constant', constant_values=(0,0))
            elif shift[0] > 0:
                shift = np.abs(shift[0])
                audio = np.pad(audio, ((15 + shift)*aud_fact, (15 - shift)*aud_fact), 'constant', constant_values=(0,0))
            elif shift[0] < 0:
                shift = np.abs(shift[0])
                audio = np.pad(audio, ((15 - shift)*aud_fact, (15 + shift)*aud_fact), 'constant', constant_values=(0,0))
            #audio = np.expand_dims(audio,axis=0)
            #video = np.expand_dims(video,axis=0)
            
            out_dict = {
                'video': video,
                'audio': audio,
                'sample': vid_path,
                'second': idx,
                'shift': true_shift,
                'category': category
            }
            out_dicts.append(out_dict)

        return out_dicts

    def __load_video__(self, vid_path, resize=None):

        frames = load_mp4(vid_path)

        if resize:
            import torchvision
            from PIL import Image
            ims = [Image.fromarray(frm) for frm in frames]
            ims = [
                torchvision.transforms.functional.resize(im,
                                                         [resize, resize], 
                                                         interpolation=InterpolationMode.BICUBIC)
                for im in ims
            ]
            frames = np.array([np.array(im) for im in ims])

        return frames.astype('float32')

    def trunkate_audio_and_video(self, video, aud_feats, aud_fact):

        aud_in_frames = aud_feats.shape[0] // aud_fact

        # make audio exactly devisible by video frames
        aud_cutoff = min(video.shape[0], int(aud_feats.shape[0] / aud_fact))

        aud_feats = aud_feats[:aud_cutoff * aud_fact]
        aud_in_frames = aud_feats.shape[0] // aud_fact

        min_len = min(aud_in_frames, video.shape[0])

        # --- trunkate all to min
        video = video[:min_len]
        aud_feats = aud_feats[:min_len * aud_fact]
        if not aud_feats.shape[0] // aud_fact == video.shape[0]:
            import ipdb
            ipdb.set_trace(context=20)

        return aud_feats, video


class FakeMix_testttt(data.Dataset):

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
            
            video = self.__load_video__(full_video_path, resize=224)
            video = video.transpose([3, 0, 1, 2])  # t h w c -> c t h w
            
            audio = load_wav(full_audio_path).astype('float32')
            aud_fact = int(np.round(16000/ 25))
            audio, video = self.trunkate_audio_and_video(video, audio, aud_fact)
            
            audio = np.array(audio)
            # audio = np.pad(audio, (15*aud_fact, 15*aud_fact), 'constant', constant_values=(0,0))
            audio = np.pad(audio, (5*aud_fact, 5*aud_fact), 'constant', constant_values=(0,0))
            
            shift = np.array([0])
            
            out_dict = {
                'video': video,
                'audio': audio,
                'sample': full_video_path,
                'shift': shift
            }
            
            return out_dict
            
            # full_video_path, full_audio_path, video_audio_class, annotations = self.samples[idx]
            # video_tensor, fps = self.process_video(full_video_path)
            # audio, sr = librosa.load(full_audio_path, sr=None)

            # video_segments_data = []
            # mfcc_segments_data = []
            # video_mfcc_segments_class = []
            # for ann in annotations:
            #     clip_start_frame, clip_end_frame = ann['frames']
            #     start_idx = min(clip_start_frame, len(video_tensor) - 1)
            #     end_idx = min(clip_end_frame + 1, len(video_tensor))
            #     clip_data_tensor = video_tensor[start_idx:end_idx]
                
            #     audio_start = clip_start_frame / fps
            #     audio_end = (clip_end_frame + 1) / fps
            #     audio_segment = audio[int(audio_start * sr):int(audio_end * sr)]
            #     mfcc_segment_tensor = self.process_audio(audio_segment, sr)
                
            #     clip_mfcc_class = self.category_to_idx_video_audio[ann['category']]
                
            #     video_segments_data.append(clip_data_tensor)
            #     mfcc_segments_data.append(mfcc_segment_tensor)
            #     video_mfcc_segments_class.append(torch.tensor(clip_mfcc_class))
            
            # return {
            #     'video_data': torch.stack(video_segments_data),
            #     'video_path': full_video_path,
            #     'mfcc_data': torch.stack(mfcc_segments_data),
            #     'mfcc_path': full_audio_path,
            #     'video_mfcc_class': torch.stack(video_mfcc_segments_class),
            # }
    
    def __load_video__(self, vid_path, resize=None):

        frames = load_mp4(vid_path)

        if resize:
            import torchvision
            from PIL import Image
            ims = [Image.fromarray(frm) for frm in frames]
            ims = [
                torchvision.transforms.functional.resize(im,
                                                         [resize, resize], 
                                                         interpolation=InterpolationMode.BICUBIC)
                for im in ims
            ]
            frames = np.array([np.array(im) for im in ims])

        return frames.astype('float32')
    
    def trunkate_audio_and_video(self, video, aud_feats, aud_fact):

        aud_in_frames = aud_feats.shape[0] // aud_fact

        # make audio exactly devisible by video frames
        aud_cutoff = min(video.shape[0], int(aud_feats.shape[0] / aud_fact))

        aud_feats = aud_feats[:aud_cutoff * aud_fact]
        aud_in_frames = aud_feats.shape[0] // aud_fact

        min_len = min(aud_in_frames, video.shape[0])

        # --- trunkate all to min
        video = video[:min_len]
        aud_feats = aud_feats[:min_len * aud_fact]
        if not aud_feats.shape[0] // aud_fact == video.shape[0]:
            import ipdb
            ipdb.set_trace(context=20)

        return aud_feats, video
    
    
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