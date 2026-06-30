# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 15:34:37 2024

@author: 12700
"""

import os
import csv
import pandas as pd

def read_labelscsv_file(csv_file_path, group):
    file_name = os.path.basename(csv_file_path)
    group_number = file_name.split('B')[1].split('M')[0].split('D')[0]
    # reverse_groups = {'10', '11', '12', '13', '14', '15', '16', '17', '20'}
    reverse_groups = {'x'}
    reverse_value = group_number in reverse_groups

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
        if reverse_value:
            ch = 'ChD' if key[:2] == 'MS' else 'ChS'
        else:
            ch = 'ChD' if key[:2] == 'MD' else 'ChS'
        if ch not in behavior_frames_dict.keys():
            behavior_frames_dict[ch] = {}
        if group == 's':
            behavior_frames_dict[ch][key] = results
        elif group == 'p':
            behavior_frames_dict[ch][key[3:]] = results
    # if group == 'p':
    #     print(file_name, len(behavior_frames_dict['ChD']['in_proximity']), len(behavior_frames_dict['ChS']['in_proximity']))
    #     if len(behavior_frames_dict['ChD']['in_proximity']) >= len(behavior_frames_dict['ChS']['in_proximity']):
    #         behavior_frames_dict['ChS']['in_proximity'] = behavior_frames_dict['ChD']['in_proximity']
    #     else:
    #         behavior_frames_dict['ChD']['in_proximity'] = behavior_frames_dict['ChS']['in_proximity']
    return behavior_frames_dict


def get_csv_files(directory):
    return [f for f in os.listdir(directory) if f.endswith('.csv')]


def create_annot_content(file_name, group):
    base_name = file_name[:-11]
    movie_file = f"{base_name}.avi"
    group_number = file_name.split('B')[1].split('M')[0]
    # if group == 's':
        # mouse_number = file_name.split('M')[1].split('D')[0]
    reverse_groups = {'1', '3', '7', '10', '11', '12', '13', '14', '15', '16', '17', '20'}
    reverse_value = group_number in reverse_groups
    
    if group == 's':
        channels = "Ch1"
        # if reverse_value:
        #     if mouse_number == '2':
        #         channels = "ChD"
        #     elif mouse_number == '1':
        #         channels = "ChS"
        # else:
        #     if mouse_number == '1':
        #         channels = "ChD"
        #     elif mouse_number == '2':
        #         channels = "ChS"
    elif group == 'p':
        channels = "ChD\nChS"
    annotations = """escape
freezing
tail_rattling
rearing/up_stretch
huddling
approach_partner
follow_partner
groom_partner
sniff_partner
grooming
sniffing
nest"""
    
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
    # print(f"annot file created at: {new_file_path}")
    

# csv_dir = r"G:\Li_lab\lst\deg_csv_data\labels_csv_socialbuffer_s"
# annot_path = r'G:\Li_lab\lst\bento_annot_data\bento_socialbuffer_s'
# group = 's'
# csv_files = get_csv_files(csv_dir)

# for csv_file in csv_files:
#     csv_file_path = os.path.join(csv_dir, csv_file)
#     behavior_frames_dict = read_labelscsv_file(csv_file_path, group)
#     annot_content = create_annot_content(csv_file, group)
#     creat_annot_file(csv_file, annot_path, annot_content, behavior_frames_dict)
