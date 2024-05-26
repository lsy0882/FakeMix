import json

def load_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def calculate_accuracy(data):
    total_n_videos = len(data)
    print(f"Number of total videos: {total_n_videos}")
    videos = list(data.values())
    clip_acc = 0
    FDM_correct = 0
    total_clips = 0
    
    for video in videos:
        clip_num = len(video)
        TA_correct = 0
        for clip in video:
            total_clips += 1
            clip_idx, category, probability = clip
            scaled_prob = 2 * probability
            if category == "RealVideo-RealAudio":
                if scaled_prob <= 0.5:
                    TA_correct += 1
                    FDM_correct += 1
            else:
                if scaled_prob >= 0.5:
                    TA_correct += 1
                    FDM_correct += 1
        clip_acc += TA_correct / clip_num
    
    TA = clip_acc / total_n_videos
    FDM = FDM_correct / total_clips
    
    return TA, FDM

file_path = '/home/lsy/laboratory/Research/FakeMix/Baseline_2_Multimodal_AVAD/tools_for_FakeMix/testing_scores_for_eval.json'

data = load_data(file_path)

TA, FDM = calculate_accuracy(data)
print(f"Temporal Accuracy (TA): {TA:.4f}")
print(f"Frame-wise Discrimination Metric (FDM): {FDM:.4f}")
