import json


file_path = "/home/lsy/laboratory/Research/idea4_MDFD/audio-visual-forensics/tools_for_FakeMix/testing_scores.json"
with open(file_path, 'r') as file:
    data = json.load(file)

organized_data = {}

for key, value in data.items():
    base_path = key.split('_sec_')[0] + '_mixed.mp4'
    sec_number = int(key.split('_sec_')[1])
    
    if base_path not in organized_data:
        organized_data[base_path] = []
    organized_data[base_path].append([sec_number] + value)

output_path = "/home/lsy/laboratory/Research/idea4_MDFD/audio-visual-forensics/tools_for_FakeMix/testing_scores_for_eval.json"
with open(output_path, 'w') as json_file:
    json.dump(organized_data, json_file, indent=4)
    
print(f"Data has been organized and saved to {output_path}")