import os
import glob

root_directory = '/home/lsy/laboratory/Research/FakeMix/data/test/'

output_file = '/home/lsy/laboratory/Research/FakeMix/Baseline_2_Multimodal_AVAD/tools_for_FakeMix/FakeMix_mp4_paths.txt'

mp4_paths = []

for dirpath, dirnames, filenames in os.walk(root_directory):
    for filename in glob.glob(os.path.join(dirpath, '*.mp4')):
        mp4_paths.append(filename)

with open(output_file, 'w') as file:
    for path in mp4_paths:
        file.write(path + '\n')