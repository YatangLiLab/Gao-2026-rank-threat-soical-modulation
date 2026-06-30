# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 15:34:37 2024

@author: 12700
"""

import os
import csv
import pandas as pd

def read_labelscsv_file(csv_file_path):
    df = pd.read_csv(csv_file_path, usecols=lambda column: column not in ['background', 'Unnamed: 0'])
    raw_data_dict = {col: df[col] for col in df.columns}
    behavior_frames_dict = {}
    for key, series in raw_data_dict.items():
        results = []
        start_idx = None
        for i in range(len(series)):
            if series[i] == 1 and start_idx is None:
                start_idx = i
            elif series[i] == 0 and start_idx is not None:
                results.append((start_idx, i - 1))
                start_idx = None
        if start_idx is not None:
            results.append((start_idx, len(series) - 1))
        if '&' in csv_file_path:
            ch = 'Channel_D' if key[:2] == 'MD' else 'Channel_S'
            if ch not in behavior_frames_dict.keys():
                behavior_frames_dict[ch] = {}
            behavior_frames_dict[ch][key[3:]] = results
        else:
            if csv_file_path.split('D')[1].split('M')[0] == '4':
                ch = 'Channel_S' if csv_file_path.split('M')[1].split('_')[0] == 1 else 'Channel_D'
            else:
                ch = 'Channel_D' if csv_file_path.split('M')[1].split('_')[0] == 1 else 'Channel_S'
            if ch not in behavior_frames_dict.keys():
                behavior_frames_dict[ch] = {}
            behavior_frames_dict[ch][key] = results
    return behavior_frames_dict


def get_csv_files(directory):
    return [f for f in os.listdir(directory) if f.endswith('.csv')]


def create_annot_content(file_name):
    base_name = file_name.split("_")[0]
    movie_file = f"{base_name}.avi"
    if '&' not in base_name:
        if base_name.split('M')[1].split('_')[0] == '1':
            channels = "Channel_D"
        elif base_name.split('M')[1].split('_')[0] == '2':
            channels = "Channel_S"
        annotations = """rat_in
approach
investigation
withdrawal
stretch-attend
freezing
tail_rattling
huddling
approach_partner
follow_partner
groom_partner
sniff_partner
grooming
rearing
sniffing"""
    elif '&' in base_name:
        channels = "Channel_D\nChannel_S"
        annotations = """rat_in
approach
investigation
withdrawal
stretch-attend
freezing
tail_rattling
huddling
approach_partner
follow_partner
groom_partner
sniff_partner
grooming
rearing
sniffing"""
    
    return f"""Bento annotation file
Movie file(s):  {movie_file}

Stimulus name: 
Annotation start time: 3.300330e-02
Annotation stop time: 6.783828e+02
Annotation framerate: 30.000000

List of channels:
{channels}

List of annotations:
{annotations}

"""

def creat_annot_file(file_name, annot_path, annot_content, behavior_frames_dict):
    for channel, behaviors in behavior_frames_dict.items():
        annot_content += f"{channel}----------\n"
        for behavior, frames in behaviors.items():
            annot_content += f">{behavior}\nStart\t Stop\t Duration \n"
            for start, stop in frames:
                duration = stop - start
                annot_content += f"{start}\t{stop}\t{duration}\n"
            annot_content += "\n"
    new_file_path = os.path.join(annot_path, file_name[:-11]+'.annot')
    with open(new_file_path, 'w') as new_file:
        new_file.write(annot_content)
    print(f"annot file created at: {new_file_path}")
    

# csv_dir = r"G:\Li_lab\ret\deg_csv_data\labels_csv_socialbuffer"
# annot_path = r'G:\Li_lab\ret\bento_ret'

# csv_files = get_csv_files(csv_dir)

# for csv_file in csv_files:
# # for csv_file in ['D10M1&M2_labels.csv']:
#     csv_file_path = os.path.join(csv_dir, csv_file)
#     behavior_frames_dict = read_labelscsv_file(csv_file_path)
#     annot_content = create_annot_content(csv_file)
#     creat_annot_file(csv_file, annot_path, annot_content, behavior_frames_dict)
