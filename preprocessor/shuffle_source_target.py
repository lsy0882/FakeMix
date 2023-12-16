import random
from moviepy.editor import VideoFileClip, concatenate_videoclips

# 영상 경로 설정
original_video_path = '/home/lsy/laboratory/Research/idea4_MDFD/sample/00073.mp4'  # 원본 영상 경로
deepfake_video_path = '/home/lsy/laboratory/Research/idea4_MDFD/sample/00073_id00137_wavtolip.mp4'  # 딥페이크 영상 경로
output_video_path = '/home/lsy/laboratory/Research/idea4_MDFD/sample/shuffle2.mp4'      # 결과 영상 경로

# 영상 파일 로드
original_clip = VideoFileClip(original_video_path)
deepfake_clip = VideoFileClip(deepfake_video_path)

# 클립 리스트 초기화
clips = []

# 영상 길이 확인 (두 영상은 길이가 동일하다고 가정)
video_length = int(original_clip.duration)

# 각 초마다 랜덤하게 클립 선택
for i in range(video_length):
    if i == 1:
        clips.append(original_clip.subclip(i, i+1))
    else:
        # Ensure that the deepfake_clip's duration is not shorter than the original clip
        if i < int(deepfake_clip.duration):
            clips.append(deepfake_clip.subclip(i, i+1))
        else:
            clips.append(original_clip.subclip(i, i+1))
        
# 각 클립의 오디오를 별도의 리스트로 추출
audio_clips = [clip.audio for clip in clips]

# 모든 클립을 이어 붙여 새로운 비디오 생성
video = concatenate_videoclips(clips, method="compose")

# 모든 오디오 클립을 이어 붙여 새로운 오디오 생성
audio = concatenate_videoclips(audio_clips, method="compose").audio

# 비디오에 오디오를 설정
final_clip = video.set_audio(audio)

# 결과 영상 저장
final_clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac')

# 사용한 리소스 해제
original_clip.close()
deepfake_clip.close()
final_clip.close()
