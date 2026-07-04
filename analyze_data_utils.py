import os
import copy
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

def lst_reorganize_data(behavior_frames_dict):
    '''Reorganize looming deepethogram labeled data.'''
    reorganized_data = {'s': {'D': {}, 'S': {}}, 'p': {'D': {}, 'S': {}}}
    for key, value in behavior_frames_dict.items():
        if 'M' in key:
            if key.split('M')[0].split('B')[1] in ['1', '3', '7', '10', '11', '12', '13', '14', '15', '16', '17', '20']:
                if key.split('M')[1].split('D')[0] == '1':
                    rank = 'S'
                elif key.split('M')[1].split('D')[0] == '2':
                    rank = 'D'
                t_id = 'B' + key.split('M')[0].split('B')[1] + 'M' + rank + 'T' + key.split('T')[1]
                reorganized_data['s'][rank][t_id] = value
            else:
                if key.split('M')[1].split('D')[0] == '1':
                    rank = 'D'
                elif key.split('M')[1].split('D')[0] == '2':
                    rank = 'S'
                t_id = 'B' + key.split('M')[0].split('B')[1] + 'M' + rank + 'T' + key.split('T')[1]
                reorganized_data['s'][rank][t_id] = value
        else:
            td_id = 'B' + key.split('D')[0].split('B')[1] + 'MDT' + key.split('T')[1]
            ts_id = 'B' + key.split('D')[0].split('B')[1] + 'MST' + key.split('T')[1]
            for sub_key, sub_value in value.items():
                if sub_key == 'ChD':
                    reorganized_data['p']['D'][td_id] = sub_value
                elif sub_key == 'ChS':
                    reorganized_data['p']['S'][ts_id] = sub_value
    return reorganized_data

def lst_read_deepethogram_csv(lst_labels_dir, bento_correction_file):
    '''Read looming deepethogram csv files, and organize it into a dict.'''
    lst_labels_dir = os.path.abspath(os.path.join('.', 'lst', 'deg_csv_data'))
    groups = ['s', 'p']
    ranks = ['D', 'S']
    behavior_frames_labels_dict = {}
    baseline_behavior_frames_labels_dict = {}
    for file_name in os.listdir(lst_labels_dir):
        if file_name.endswith('.csv'):
            file_path = os.path.join(lst_labels_dir, file_name)
            data = pd.read_csv(file_path, index_col=0, usecols=lambda column: column != 'background')
            headers = list(data.columns)
            content = data.values
            if 'baseline' not in file_name:
                behavior_frames_labels_dict[file_name.split('_')[-2]] = {}
                if 'M' not in file_name:
                    for header in headers:
                        key_base = 'ChD' if header.startswith('MD_') else 'ChS'
                        key = header[3:]
                        if key_base not in behavior_frames_labels_dict[file_name.split('_')[-2]]:
                            behavior_frames_labels_dict[file_name.split('_')[-2]][key_base] = {}
                        behavior_frames_labels_dict[file_name.split('_')[-2]][key_base][key] = data[header]
                elif 'M'  in file_name:
                    for header in headers:
                        behavior_frames_labels_dict[file_name.split('_')[-2]][header] = data[header]

            elif 'baseline' in file_name:
                baseline_behavior_frames_labels_dict[file_name.split('_')[-2]] = {}
                if 'M' not in file_name:
                    for header in headers:
                        key_base = 'ChD' if header.startswith('MD_') else 'ChS'
                        key = header[3:]
                        if key_base not in baseline_behavior_frames_labels_dict[file_name.split('_')[-2]]:
                            baseline_behavior_frames_labels_dict[file_name.split('_')[-2]][key_base] = {}
                        baseline_behavior_frames_labels_dict[file_name.split('_')[-2]][key_base][key] = data[header]
                elif 'M'  in file_name:
                    for header in headers:
                        baseline_behavior_frames_labels_dict[file_name.split('_')[-2]][header] = data[header]

    for key in behavior_frames_labels_dict.keys():
        if key not in baseline_behavior_frames_labels_dict:
            original_key = key.split('T')[0] + 'T1'  # assuming the original key to copy from has the same prefix
            if original_key in baseline_behavior_frames_labels_dict:
                baseline_behavior_frames_labels_dict[key] = copy.deepcopy(baseline_behavior_frames_labels_dict[original_key])

    correction_data = pd.read_csv(bento_correction_file, header=None, usecols=[0, 1], nrows=340)  # real looming start frame
    correction_data[1] -= 150  # Considering that the video starts from -5s
    correction_dict = dict(zip(correction_data[0], correction_data[1]))
    for key in behavior_frames_labels_dict.keys():  # correct behavior_frames_labels_dict
        if key in correction_dict:
            correction_value = correction_dict[key]
            for sub_key in behavior_frames_labels_dict[key]:
                if sub_key in ['ChS', 'ChD']:
                    for sub_sub_key in behavior_frames_labels_dict[key][sub_key]:
                        original_series = behavior_frames_labels_dict[key][sub_key][sub_sub_key]
                        corrected_series = original_series.shift(-correction_value)
                        corrected_series = corrected_series[corrected_series.index >= 0]
                        behavior_frames_labels_dict[key][sub_key][sub_sub_key] = corrected_series
                else:
                    original_series = behavior_frames_labels_dict[key][sub_key]
                    corrected_series = original_series.shift(-correction_value)
                    corrected_series = corrected_series[corrected_series.index >= 0]
                    behavior_frames_labels_dict[key][sub_key] = corrected_series


    behavior_frames_labels_dict = lst_reorganize_data(behavior_frames_labels_dict)
    baseline_behavior_frames_labels_dict = lst_reorganize_data(baseline_behavior_frames_labels_dict)

    correction_dict = dict(zip(correction_data[0], correction_data[1]))
    for key in behavior_frames_labels_dict.keys():  # correct behavior_frames_labels_dict
        if key in correction_dict:
            correction_value = correction_dict[key]
            for sub_key in behavior_frames_labels_dict[key]:
                original_series = behavior_frames_labels_dict[key][sub_key]
                corrected_series = original_series.shift(-correction_value)
                corrected_series = corrected_series[corrected_series.index >= 0]
                behavior_frames_labels_dict[key][sub_key] = corrected_series

    return behavior_frames_labels_dict, baseline_behavior_frames_labels_dict


def lst_read_annotation_file(file_path):
    ''' Read bento annot file.'''
    behavior_frames = {}
    behaviors = [
        'approach_partner',
        'climbing',
        'dwelling',
        'escape',
        'follow_partner', 
        'freezing',
        'groom_partner', 
        'grooming',
        'jumping',
        'nest',
        'reaction',
        'looming',
        'rearing/up_stretch',
        'real_rearing', 
        'sniff_partner',
        'sniffing',
        'stretching', 
        'tailrattling']
    # initialize behavior_frames with all possible behaviors
    for behavior in behaviors:
        behavior_frames[behavior] = []
    file_name = os.path.basename(file_path)
    if 'M' in file_name:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        current_behavior = None
        for line in lines:
            if line.startswith('>'):
                current_behavior = line.strip()[1:]
                behavior_frames[current_behavior] = []
            elif line.strip() != '':
                try:
                    start_frame, stop_frame, _ = map(int, line.split())
                    behavior_frames[current_behavior].append((start_frame, stop_frame))
                except ValueError:
                    pass
    else:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        chd_dict = {}
        chs_dict = {}
        current_dict = {}
        current_behavior = None
        for behavior in behaviors:
            chd_dict[behavior] = []
            chs_dict[behavior] = []
        for line in lines:
            if line.startswith('ChD----------'):
                current_dict = chd_dict
            elif line.startswith('ChS----------'):
                current_dict = chs_dict
            elif line.strip() != '':
                if line.startswith('>'):
                    current_behavior = line.strip()[1:]
                    current_dict[current_behavior] = []
                else:
                    try:
                        start_frame, stop_frame, _ = map(int, line.split())
                        current_dict[current_behavior].append((start_frame, stop_frame))
                    except ValueError:
                        pass
        behavior_frames = {'ChD': chd_dict, 'ChS': chs_dict}
    return behavior_frames


def lst_correct_bento_data(behavior_frames_dict, correction_data):
    '''Apply the modifications to the bento data sequence.'''
    corrected_dict = behavior_frames_dict.copy()
    for key, value in behavior_frames_dict.items():
        for sub_key, sub_value in value.items():
            if isinstance(sub_value, list):
                correction_row = correction_data[correction_data[0] == key]
                correction_value = correction_row.iloc[0, 1]
                corrected_frames = []
                for start_frame, stop_frame in sub_value:
                    corrected_frames.append((start_frame - correction_value, stop_frame - correction_value))
                corrected_dict[key][sub_key] = corrected_frames
            else:
                for sub_sub_key, sub_sub_value in sub_value.items():
                    correction_row = correction_data[correction_data[0] == key]
                    correction_value = correction_row.iloc[0, 1]
                    corrected_frames = []
                    for start_frame, stop_frame in sub_sub_value:
                        corrected_frames.append((start_frame - correction_value, stop_frame - correction_value))
                    corrected_dict[key][sub_key][sub_sub_key] = corrected_frames
    return corrected_dict


def lst_read_bento_annot(lst_bento_dir, lst_bento_corr_file, lst_deci_dict):
    '''Read bento annot files, and organize it into a dict.'''
    behavior_frames_dict = {}  # all bento data
    behavior_frames_dict_p = {}  # pair bento data, {'BXDXTX': {'ChD: sub_dict, 'ChS': sub_dict}}
    behavior_frames_dict_s = {}  # single bento data, {'BXMXDXTX': sub_dict}
    per_behavior_frames_dict = {}
    per_behavior_frames_dict_p = {}
    per_behavior_frames_dict_s = {}
    baseline_behavior_frames_dict = {}
    baseline_behavior_frames_dict_p = {}
    baseline_behavior_frames_dict_s = {}
    for file_name in os.listdir(lst_bento_dir):
        if file_name.endswith('.annot'):
            t_id = file_name[:-6]
            file_path = os.path.join(lst_bento_dir, file_name)
            if t_id.startswith('per_'):
                per_behavior_frames = lst_read_annotation_file(file_path)
                per_behavior_frames_dict[t_id] = per_behavior_frames
            elif t_id.startswith('baseline_'):
                baseline_behavior_frames = lst_read_annotation_file(file_path)
                baseline_behavior_frames_dict[t_id] = baseline_behavior_frames
            else:
                behavior_frames = lst_read_annotation_file(file_path)
                behavior_frames_dict[t_id] = behavior_frames
    # print(behavior_frames_dict.keys())
    correction_data = pd.read_csv(lst_bento_corr_file, header=None, usecols=[0, 1], nrows=340)  # real looming start frame
    correction_data[1] -= 150
    corrected_behavior_frames_dict = lst_correct_bento_data(behavior_frames_dict, correction_data)  # correct frame
    per_correction_data = pd.read_csv(lst_bento_corr_file, header=None, usecols=[0, 1], skiprows=340)  # real looming start frame
    per_correction_data[1] -= 930
    corrected_per_behavior_frames_dict = lst_correct_bento_data(per_behavior_frames_dict, per_correction_data)  # correct frame

    # print(corrected_behavior_frames_dict)
    for key, value in corrected_behavior_frames_dict.items():
        if 'M' not in key:
            behavior_frames_dict_p[key] = value
        else:
            behavior_frames_dict_s[key] = value
    for key, value in corrected_per_behavior_frames_dict.items():
        if 'M' not in key:
            per_behavior_frames_dict_p[key] = value
        else:
            per_behavior_frames_dict_s[key] = value
    for key, value in baseline_behavior_frames_dict.items():
        if 'M' not in key:
            baseline_behavior_frames_dict_p[key] = value
        else:
            baseline_behavior_frames_dict_s[key] = value

    behavior_frames_dict_p_new = {}  # this dict is the same as the structure of behavior_frames_dict_s
    for key, value in behavior_frames_dict_p.items():
        if key.split('B')[1].split('D')[0] in ['1', '3', '7', '10', '11', '12', '13', '14', '15', '16', '17', '20']:
            new_key_1 = key.replace(key.split('D')[0], key.split('D')[0]+'M2')
            new_key_2 = key.replace(key.split('D')[0], key.split('D')[0]+'M1')
        else:
            new_key_1 = key.replace(key.split('D')[0], key.split('D')[0]+'M1')
            new_key_2 = key.replace(key.split('D')[0], key.split('D')[0]+'M2')
        behavior_frames_dict_p_new[new_key_1] = value['ChD']
        behavior_frames_dict_p_new[new_key_2] = value['ChS']
    # per_behavior_frames_dict_p_new = {} 
    # for key, value in per_behavior_frames_dict_p.items():
    #     if key.split('B')[1].split('D')[0] in ['1', '3', '7', '10', '11', '12', '13', '14', '15', '16', '17', '20']:
    #         new_key_1 = key.replace(key.split('D')[0], key.split('D')[0]+'M2')
    #         new_key_2 = key.replace(key.split('D')[0], key.split('D')[0]+'M1')
    #     else:
    #         new_key_1 = key.replace(key.split('D')[0], key.split('D')[0]+'M1')
    #         new_key_2 = key.replace(key.split('D')[0], key.split('D')[0]+'M2')
    #     per_behavior_frames_dict_p_new[new_key_1] = value['ChD']
    #     per_behavior_frames_dict_p_new[new_key_2] = value['ChS']
    baseline_behavior_frames_dict_p_new = {} 
    for key, value in baseline_behavior_frames_dict_p.items():
        if key.split('B')[1].split('D')[0] in ['1', '3', '7', '10', '11', '12', '13', '14', '15', '16', '17', '20']:
            new_key_1 = key.replace(key.split('D')[0], key.split('D')[0]+'M2')
            new_key_2 = key.replace(key.split('D')[0], key.split('D')[0]+'M1')
        else:
            new_key_1 = key.replace(key.split('D')[0], key.split('D')[0]+'M1')
            new_key_2 = key.replace(key.split('D')[0], key.split('D')[0]+'M2')
        baseline_behavior_frames_dict_p_new[new_key_1] = value['ChD']
        baseline_behavior_frames_dict_p_new[new_key_2] = value['ChS']
    behavior_frames_dict = lst_reorganize_data(corrected_behavior_frames_dict)
    print(behavior_frames_dict['s']['D'].keys())
    # per_behavior_frames_dict = reorganize_data(corrected_per_behavior_frames_dict)
    # print(per_behavior_frames_dict['s']['D'].keys())
    baseline_behavior_frames_dict = lst_reorganize_data(baseline_behavior_frames_dict)
    print(baseline_behavior_frames_dict['p']['D'].keys())

    for key, value in lst_deci_dict.items():
        for sub_key, sub_value in value.items():
            for sub_sub_key, sub_sub_value in sub_value.items():
                if sub_sub_value == 'N' or sub_sub_value == 'n':
                    if key in behavior_frames_dict and sub_key in behavior_frames_dict[key]:
                        del behavior_frames_dict[key][sub_key][sub_sub_key]
                        # del per_behavior_frames_dict[key][sub_key][sub_sub_key]
    print(behavior_frames_dict['p']['D'].keys())
    # print(per_behavior_frames_dict['p']['D'].keys())

    return behavior_frames_dict, baseline_behavior_frames_dict

def lst_read_deeplabcut_h5_and_analyze_data(lst_dlc_h5_dir):
    '''Read deeplabcut h5 files, analyze data and organize it into a dict.'''
    all_h5_files = [os.path.join(lst_dlc_h5_dir, f) for f in os.listdir(lst_dlc_h5_dir) if f.endswith('.h5')]
    
    # lts_h5_files = sorted([f for f in all_h5_files if 'DLC_resnet50_Looming_MouseSep13shuffle1_400000' in f])
    # ltp_h5_files = sorted([f for f in all_h5_files if 'DLC_dlcrnetms5_Looming_MiceJan19shuffle1_200000_el' in f])
    
    lts_h5_files = [
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20230713100856143DLC_resnet50_Looming_MouseSep13shuffle1_400000.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20230713105013341DLC_resnet50_Looming_MouseSep13shuffle1_400000.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20230713120627186DLC_resnet50_Looming_MouseSep13shuffle1_400000.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20230713124041145DLC_resnet50_Looming_MouseSep13shuffle1_400000.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20230713131420679DLC_resnet50_Looming_MouseSep13shuffle1_400000.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20230713134532734DLC_resnet50_Looming_MouseSep13shuffle1_400000.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20230817131809396DLC_resnet50_Looming_MouseSep13shuffle1_400000.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20230817140239139DLC_resnet50_Looming_MouseSep13shuffle1_400000.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20230818093613403DLC_resnet50_Looming_MouseSep13shuffle1_400000.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20230818100756250DLC_resnet50_Looming_MouseSep13shuffle1_400000.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20230822093007876DLC_resnet50_Looming_MouseSep13shuffle1_400000.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20230822100844426DLC_resnet50_Looming_MouseSep13shuffle1_400000.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20230822104920465DLC_resnet50_Looming_MouseSep13shuffle1_400000.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20230822114248055DLC_resnet50_Looming_MouseSep13shuffle1_400000.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20230822124940813DLC_resnet50_Looming_MouseSep13shuffle1_400000.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20230822133227492DLC_resnet50_Looming_MouseSep13shuffle1_400000.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20240820213608378DLC_resnet50_Looming_MouseSep13shuffle1_400000.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20240820222021991DLC_resnet50_Looming_MouseSep13shuffle1_400000.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20240820230212590DLC_resnet50_Looming_MouseSep13shuffle1_400000.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20240820234051059DLC_resnet50_Looming_MouseSep13shuffle1_400000.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20240821001548097DLC_resnet50_Looming_MouseSep13shuffle1_400000.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20240821004731168DLC_resnet50_Looming_MouseSep13shuffle1_400000.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20240821024325105DLC_resnet50_Looming_MouseSep13shuffle1_400000.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20240821033652998DLC_resnet50_Looming_MouseSep13shuffle1_400000.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20240821041633645DLC_resnet50_Looming_MouseSep13shuffle1_400000.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20240821045124283DLC_resnet50_Looming_MouseSep13shuffle1_400000.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20240821053354009DLC_resnet50_Looming_MouseSep13shuffle1_400000.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20240821060922927DLC_resnet50_Looming_MouseSep13shuffle1_400000.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20240823213821672DLC_resnet50_Looming_MouseSep13shuffle1_400000.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20240823204456512DLC_resnet50_Looming_MouseSep13shuffle1_400000.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20240823232706501DLC_resnet50_Looming_MouseSep13shuffle1_400000.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20240823223246767DLC_resnet50_Looming_MouseSep13shuffle1_400000.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20240824011629202DLC_resnet50_Looming_MouseSep13shuffle1_400000.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20240824002146995DLC_resnet50_Looming_MouseSep13shuffle1_400000.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20240824030530481DLC_resnet50_Looming_MouseSep13shuffle1_400000.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20240824021130041DLC_resnet50_Looming_MouseSep13shuffle1_400000.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20240824074434610DLC_resnet50_Looming_MouseSep13shuffle1_400000.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20240824040020200DLC_resnet50_Looming_MouseSep13shuffle1_400000.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20240824123520595DLC_resnet50_Looming_MouseSep13shuffle1_400000.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20240824084140333DLC_resnet50_Looming_MouseSep13shuffle1_400000.h5"]
    ltp_h5_files = [
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20230713142319732DLC_dlcrnetms5_Looming_MiceJan19shuffle1_200000_el.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20230713152610177DLC_dlcrnetms5_Looming_MiceJan19shuffle1_200000_el.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20230713162336311DLC_dlcrnetms5_Looming_MiceJan19shuffle1_200000_el.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20230817121500446DLC_dlcrnetms5_Looming_MiceJan19shuffle1_200000_el.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20230818105841630DLC_dlcrnetms5_Looming_MiceJan19shuffle1_200000_el.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20230821122406377DLC_dlcrnetms5_Looming_MiceJan19shuffle1_200000_el.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20230821132318983DLC_dlcrnetms5_Looming_MiceJan19shuffle1_200000_el.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20230821142223165DLC_dlcrnetms5_Looming_MiceJan19shuffle1_200000_el.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20240821230157875DLC_dlcrnetms5_Looming_MiceJan19shuffle1_200000_el.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20240821235738008DLC_dlcrnetms5_Looming_MiceJan19shuffle1_200000_el.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20240822005134869DLC_dlcrnetms5_Looming_MiceJan19shuffle1_200000_el.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20240822013602322DLC_dlcrnetms5_Looming_MiceJan19shuffle1_200000_el.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20240822022337624DLC_dlcrnetms5_Looming_MiceJan19shuffle1_200000_el.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20240822033726180DLC_dlcrnetms5_Looming_MiceJan19shuffle1_200000_el.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20240822204948647DLC_dlcrnetms5_Looming_MiceJan19shuffle1_200000_el.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20240822214330740DLC_dlcrnetms5_Looming_MiceJan19shuffle1_200000_el.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20240822223703777DLC_dlcrnetms5_Looming_MiceJan19shuffle1_200000_el.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20240822233303479DLC_dlcrnetms5_Looming_MiceJan19shuffle1_200000_el.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20240823002038828DLC_dlcrnetms5_Looming_MiceJan19shuffle1_200000_el.h5", 
        "G:\Li_lab\paper_sb\lst\dlc_h5_data\Video_20240823011337977DLC_dlcrnetms5_Looming_MiceJan19shuffle1_200000_el.h5"]
    print(f"Found {len(lts_h5_files)} lts h5")
    print(f"Found {len(ltp_h5_files)} ltp h5")

    ms_ids = ['B1M1(S)', 'B1M2(D)', 'B2M1(D)', 'B2M2(S)', 'B3M1(S)', 'B3M2(D)', 'B4M1(D)', 'B4M2(S)', 
              'B5M1(D)', 'B5M2(S)', 'B6M1(D)', 'B6M2(S)', 'B7M1(S)', 'B7M2(D)', 'B8M1(D)', 'B8M2(S)', 
            'B9M1(D)', 'B9M2(S)', 'B10M1(S)', 'B10M2(D)', 'B11M1(S)', 'B11M2(D)', 'B12M1(S)', 'B12M2(D)', 
                'B13M1(S)', 'B13M2(D)', 'B14M1(S)', 'B14M2(D)', 'B15M1(S)', 'B15M2(D)', 'B16M1(S)', 'B16M2(D)',
                'B17M1(S)', 'B17M2(D)', 'B18M1(D)', 'B18M2(S)', 'B19M1(D)', 'B19M2(S)', 'B20M1(S)', 'B20M2(D)']
    mp_ids = ['B1M1(S)_B1M2(D)', 'B2M1(D)_B2M2(S)', 'B3M1(S)_B3M2(D)', 'B4M1(D)_B4M2(S)', 
            'B5M1(D)_B5M2(S)', 'B6M1(D)_B6M2(S)', 'B7M1(S)_B7M2(D)', 'B8M1(D)_B8M2(S)', 
            'B9M1(D)_B9M2(S)', 'B10M1(S)_B10M2(D)', 'B11M1(S)_B11M2(D)', 'B12M1(S)_B12M2(D)', 
            'B13M1(S)_B13M2(D)', 'B14M1(S)_B14M2(D)', 'B15M1(S)_B15M2(D)', 'B16M1(S)_B16M2(D)', 
            'B17M1(S)_B17M2(D)', 'B18M1(D)_B18M2(S)', 'B19M1(D)_B19M2(S)', 'B20M1(S)_B20M2(D)']
    lsts_dict = get_analyzed_data_lsts(lts_h5_files, ms_ids)
    lstp_dict = get_analyzed_data_lstp(ltp_h5_files, mp_ids)
    return lsts_dict, lstp_dict


def get_data_from_hdf5_lstp(h5_file):
    '''Read h5 file for paired mouse condition.'''
    keypoints = ['nose', 'leftear', 'rightear', 'neck', 'waist', 'tail1']
    df = pd.read_hdf(h5_file)
    scorer = df.columns.get_level_values('scorer')[0]
    df_data = df[scorer]

    keypoints_dict = {}
    for i in ['M1', 'M2']:
        keypoints_dict[i] = {}
        for keypoint in keypoints:
            keypoints_dict[i][keypoint] = {
                'x': df_data[i][keypoint]['x'],
                'y': df_data[i][keypoint]['y'],
                'likelihood': df_data[i][keypoint]['likelihood']
            }
    return keypoints_dict

def get_data_from_hdf5_lsts(h5_file):
    '''Read h5 file for single mouse condition.'''
    keypoints = ['nose', 'leftear', 'rightear', 'neck', 'waist', 'tail1']
    df = pd.read_hdf(h5_file)
    scorer = df.columns.get_level_values('scorer')[0]
    df_data = df[scorer]

    keypoints_dict = {}
    for keypoint in keypoints:
        keypoints_dict[keypoint] = {
            'x': df_data[keypoint]['x'],
            'y': df_data[keypoint]['y'],
            'likelihood': df_data[keypoint]['likelihood']
        }
    return keypoints_dict


def filter_data(raw_data, threshold):
    keypoints = ['nose', 'leftear', 'rightear', 'neck', 'waist', 'tail1']
    filtered_data = {}
    for keypoint in keypoints:
        x_data = raw_data[keypoint]['x']
        y_data = raw_data[keypoint]['y']
        filled_x_data = fill_data(x_data)
        filled_y_data = fill_data(y_data)
        likelihood_data = raw_data[keypoint]['likelihood']
        
        # filter the keypoints that differ greater than 50 pix
        # dx = filled_x_data.diff(periods=1)
        # dy = filled_y_data.diff(periods=1)
        # distance = np.sqrt(dx ** 2 + dy ** 2)
        # filter_x_data = np.where(distance < 80, filled_x_data, np.nan)
        # filter_y_data = np.where(distance < 80, filled_y_data, np.nan)
        
        # rolling().mean() or .median()
        # rolling_window_x = filled_x_data.rolling(window=11, center=True).median()
        # rolling_window_y = filled_x_data.rolling(window=11, center=True).median()
        # filter_x_data = np.where(np.logical_and(rolling_window_x - 300 < filled_x_data, filled_x_data < rolling_window_x + 300), filled_x_data, np.nan)
        # filter_y_data = np.where(np.logical_and(rolling_window_y - 300 < filled_y_data, filled_y_data < rolling_window_y + 300), filled_y_data, np.nan)
        
        filtered_x_data = np.where(likelihood_data > threshold, filled_x_data, np.nan)
        filtered_y_data = np.where(likelihood_data > threshold, filled_y_data, np.nan)
        filtered_data[keypoint] = {
            'x': pd.Series(filtered_x_data), 
            'y': pd.Series(filtered_y_data)
        }
    return filtered_data


def fill_data(raw_data):
    keypoints = ['nose', 'leftear', 'rightear', 'neck', 'waist', 'tail1']
    filled_data = {}
    max_level = count_nested_layers(raw_data)
    if max_level == 2:
        for keypoint in keypoints:
            filled_data[keypoint] = {}
            for i in ['x', 'y']:
                series = raw_data[keypoint][i]
                interpolated_series = series.interpolate(method='linear')
                interpolated_series = interpolated_series.interpolate(method='pad')
                interpolated_series = interpolated_series.interpolate(method='backfill')
                filled_data[keypoint][i] = interpolated_series
    elif max_level == 1:
        for i in ['x', 'y']:
            series = raw_data[i]
            interpolated_series = series.interpolate(method='linear')
            interpolated_series = interpolated_series.interpolate(method='pad')
            interpolated_series = interpolated_series.interpolate(method='backfill')
            filled_data[i] = interpolated_series
    else:
        series = raw_data
        interpolated_series = series.interpolate(method='linear')
        interpolated_series = interpolated_series.interpolate(method='pad')
        interpolated_series = interpolated_series.interpolate(method='backfill')
        filled_data = interpolated_series
    return filled_data


def count_nested_layers(data, current_level=0):
    if isinstance(data, dict):
        max_nested_level = 1
        for key, value in data.items():
            if isinstance(value, dict):
                max_nested_level = 2
        return max_nested_level
    else:
        return current_level


def get_centroid_point(kps):
    x_values = [kp['x'] for kp in kps]
    y_values = [kp['y'] for kp in kps]

    if x_values and y_values:
        df_x = pd.DataFrame(x_values)
        centroid_x = df_x.mean(axis=0)
        df_y = pd.DataFrame(y_values)
        centroid_y = df_y.mean(axis=0)
        centroid = {'x': centroid_x, 'y': centroid_y}
        return centroid
    else:
        return None


def smooth_data(data, window_size=11):
    smoothed_data = {}
    for key, value in data.items():
        if isinstance(value, dict):
            smoothed_data[key] = {}
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, pd.Series):
                    smoothed_data[key][sub_key] = sub_value.rolling(window=window_size, center=True).mean()
                else:
                    smoothed_data[key][sub_key] = sub_value
        else:
            smoothed_data[key] = value
    return smoothed_data


def get_location_value(kp):
    nest_crood = [0, 740]
    nest_size = [480, 618]
    trigger_size = [392, 392]
    trigger_crood = [(1088 - trigger_size[0]) / 2, (1088 * 0.6 - trigger_size[1]) / 2]
    exploration_y_crood = 652
    location_value = pd.Series(np.ones_like(kp['x']))
    nest_zone_idx = np.where((kp['x'] >= nest_crood[0]) & 
                             (kp['x'] <= nest_crood[0] + nest_size[0]) & 
                             (kp['y'] >= nest_crood[1]) & 
                             (kp['y'] <= nest_crood[1] + nest_size[1]))[0]
    trigger_zone_idx = np.where((kp['x'] >= trigger_crood[0]) & 
                                (kp['x'] <= trigger_crood[0] + trigger_size[0]) & 
                                (kp['y'] >= trigger_crood[1]) & 
                                (kp['y'] <= trigger_crood[1] + trigger_size[1]))[0]
    dark_zone_idx = np.where(kp['y'] >= exploration_y_crood)[0]
    light_zone_idx = np.where(kp['y'] < exploration_y_crood)[0]
    location_value[light_zone_idx] = 2
    location_value[dark_zone_idx] = 1
    location_value[trigger_zone_idx] = 3
    location_value[nest_zone_idx] = 0
    return location_value


def get_location_value_bl(kp):
    pixpergrid = 43.52
    nest_crood = [0, pixpergrid*17]
    nest_size = [pixpergrid*10, pixpergrid*7.5]
    danger1_crood = [pixpergrid*4, pixpergrid*4]
    danger1_size = [pixpergrid*17, pixpergrid*13]
    danger2_crood = [pixpergrid*10, pixpergrid*17]
    danger2_size = [pixpergrid*4, pixpergrid*11]
    
    location_value = pd.Series(np.ones_like(kp['x']))
    nest_zone_idx = np.where((kp['x'] >= nest_crood[0]) & 
                             (kp['x'] <= nest_crood[0] + nest_size[0]) & 
                             (kp['y'] >= nest_crood[1]) & 
                             (kp['y'] <= nest_crood[1] + nest_size[1]))[0]
    danger1_zone_idx = np.where((kp['x'] >= danger1_crood[0]) & 
                                (kp['x'] <= danger1_crood[0] + danger1_size[0]) & 
                                (kp['y'] >= danger1_crood[1]) & 
                                (kp['y'] <= danger1_crood[1] + danger1_size[1]))[0]
    danger2_zone_idx = np.where((kp['x'] >= danger2_crood[0]) & 
                                (kp['x'] <= danger2_crood[0] + danger2_size[0]) & 
                                (kp['y'] >= danger2_crood[1]) & 
                                (kp['y'] <= danger2_crood[1] + danger2_size[1]))[0]
    location_value[danger1_zone_idx] = 3
    location_value[danger2_zone_idx] = 2
    location_value[nest_zone_idx] = 0
    return location_value


def get_analyzed_data_lsts(h5_files, ms_ids):
    keypoints = ['nose', 'leftear', 'rightear', 'neck', 'waist', 'tail1']
    pattern = re.compile(r'B(\d+)M(\d+)\((\w)\)')
    
    m_ids = []
    feature_data = {}
    i = -1
    for h5_file in h5_files:
        i += 1
        match = pattern.search(ms_ids[i])
        if match:
            m_id = 'B' + match.group(1) + 'M' + match.group(2) + '(' + match.group(3) + ')'
            m_ids.append(m_id)
        else:
            print('Check if the regular expression matches the h5_files.')
        raw_keypoints_data = get_data_from_hdf5_lsts(h5_file)
        filter_keypoints_data = filter_data(raw_keypoints_data, threshold=0.9)
        filled_keypoints_data = fill_data(filter_keypoints_data)

        
        trunk_keypoints = [filled_keypoints_data['nose'], 
                           filled_keypoints_data['neck'], 
                           filled_keypoints_data['waist'],
                           filled_keypoints_data['tail1']]
        centroid = get_centroid_point(trunk_keypoints)
        filled_keypoints_data['centroid'] = centroid
        
        # get test time.
        index = filled_keypoints_data['nose']['x'].index
        bins = [0, 18000, len(filled_keypoints_data['nose']['x']) - 18000, len(filled_keypoints_data['nose']['x'])]
        labels = [0, 2, 1]
        time_value = pd.cut(index, bins=bins, labels=labels, right=False, ordered=False)

        velocity_data = {}
        acceleration_data = {}
        avg_velocity_data = {}
        avg_acceleration_data = {}
        distance_traveled = {}
        for keypoint in keypoints:
            velocities, accelerations = calculate_keypoint(filled_keypoints_data[keypoint])
            filter_velocities = velocities.where(velocities < 216)
            filled_velocities = fill_data(filter_velocities)
            velocity_data[keypoint] = filled_velocities
            acceleration_data[keypoint] = accelerations
            avg_velocity_data[keypoint] = pd.Series(velocity_data[keypoint]).rolling(window=11, center=True).mean()
            avg_acceleration_data[keypoint] = pd.Series(acceleration_data[keypoint]).rolling(window=11, center=True).mean()
        avg_velocity_data['all'] = np.mean(velocity_data['waist'])
        avg_acceleration_data['all'] = np.mean(acceleration_data['waist'])
        avg_velocity_data['baseline'] = velocity_data['waist'][time_value==0]
        avg_velocity_data['withrat'] = velocity_data['waist'][time_value==1]
        cv_velocity_data = {}
        cv_velocity_data['all'] = np.std(velocity_data['waist']) / np.mean(velocity_data['waist'])
        cv_velocity_data['baseline'] = np.std(velocity_data['waist'][time_value==0]) / np.mean(velocity_data['waist'][time_value==0])
        cv_velocity_data['withrat'] = np.std(velocity_data['waist'][time_value==1]) / np.mean(velocity_data['waist'][time_value==1])
        
        nose_data_1 = {'x': filled_keypoints_data['nose']['x'][time_value == 1], 
                        'y': filled_keypoints_data['nose']['y'][time_value == 1]}
        neck_data_1 = {'x': filled_keypoints_data['neck']['x'][time_value == 1], 
                        'y': filled_keypoints_data['neck']['y'][time_value == 1]}
        waist_data = {'x': filled_keypoints_data['waist']['x'], 
                            'y': filled_keypoints_data['waist']['y']}
        distance_traveled = sum_distance_traveled(waist_data)
        centroid_data_0 = {'x': filled_keypoints_data['centroid']['x'][time_value == 0], 
                           'y': filled_keypoints_data['centroid']['y'][time_value == 0]}
        
        location4lst = get_location_value(filled_keypoints_data['centroid'])
        location4lstbl = get_location_value_bl(filled_keypoints_data['centroid'])
        # print(location4ret.value_counts().get(0, 0), len(location4ret))
        zone_framecount = {'trigger_zone': location4lst.value_counts().get(3, 0),
                           'light_zone': location4lst.value_counts().get(2, 0),
                           'dark_zone': location4lst.value_counts().get(1, 0),
                           'nest_zone': location4lst.value_counts().get(0, 0)}
        
        immobile_framecount = {'all': np.sum(np.array(velocity_data['waist']) <= 1),
                               'baseline': np.sum(np.array(velocity_data['waist'][time_value == 0]) <= 1),
                               'withrat': np.sum(np.array(velocity_data['waist'][time_value == 1]) <= 1)}

        head_angle = calculate_angle(nose_data_1, neck_data_1, 1100, method='ppl')
        # head_angle_framecount = {'le90': len(head_angle[head_angle <= 90]),
        #                          'gt90': len(head_angle[head_angle > 90])}
        
        trunk_keypoints = [filled_keypoints_data['nose'], 
                           filled_keypoints_data['neck'], 
                           filled_keypoints_data['waist'],
                           filled_keypoints_data['tail1']]
        trunk_length = calculate_distance_sum(trunk_keypoints)
        
        feature_data[m_id] = {
            'coordinate': filled_keypoints_data,
            'velocity': velocity_data,
            'acceleration': acceleration_data,
            'avg_velocity': avg_velocity_data,
            'cv_velocity':cv_velocity_data, 
            'avg_acceleration': avg_acceleration_data,
            'time_value': time_value,
            'distance_traveled': distance_traveled,
            'location4lst': location4lst,
            'location4lstbl': location4lstbl, 
            'zone_framecount': zone_framecount,
            'immobile_framecount': immobile_framecount,
            'head_angle': head_angle,
            'trunk_length': trunk_length
            }
    lsts_data = feature_data
    return lsts_data


def get_analyzed_data_lstp(h5_files, mp_ids):
    keypoints = ['nose', 'leftear', 'rightear', 'neck', 'waist', 'tail1']
    pattern = re.compile(r'B(\d+)M(\d+)\((\w)\)_B(\d+)M(\d+)\((\w)\)')

    m_ids = []
    feature_data = {}
    i = -1
    for h5_file in h5_files:
        i += 1
        match = pattern.search(mp_ids[i])
        if match:
            m_id_1 = 'B' + match.group(1) + 'M' + match.group(2) + '(' + match.group(3) + ')'
            m_id_2 = 'B' + match.group(4) + 'M' + match.group(5) + '(' + match.group(6) + ')'
            m_ids.extend([m_id_1, m_id_2])
        else:
            print('Check if the regular expression matches the h5_files.')
        raw_keypoints_data = get_data_from_hdf5_lstp(h5_file)
        new_keypoints_data = {m_id_1: raw_keypoints_data.get('M1', {}),
                              m_id_2: raw_keypoints_data.get('M2', {})}
        
        nose_data_0 = {}
        nose_data_1 = {}
        neck_data_1 = {}
        waist_data_0 = {}
        waist_data_1 = {}
        waist_data = {}
        for m_id in [m_id_1, m_id_2]:
            if m_id[1] == '1' or m_id[1] == '2' or m_id[1] == '3':
                filter_keypoints_data = filter_data(new_keypoints_data[m_id], threshold=0.8)
            else:
                if m_id == m_id_1:
                    filter_keypoints_data = filter_data(new_keypoints_data[m_id_2], threshold=0.8)
                if m_id == m_id_2:
                    filter_keypoints_data = filter_data(new_keypoints_data[m_id_1], threshold=0.8)
            filled_keypoints_data = fill_data(filter_keypoints_data)
            # smoothed_keypoints_data = smooth_data(filled_keypoints_data)
            # filled_keypoints_data = smoothed_keypoints_data
            # get the centroid point
            trunk_keypoints = [filled_keypoints_data['nose'], 
                               filled_keypoints_data['neck'], 
                               filled_keypoints_data['waist'],
                               filled_keypoints_data['tail1']]
            centroid = get_centroid_point(trunk_keypoints)
            filled_keypoints_data['centroid'] = centroid
        
            # get test time.
            index = filled_keypoints_data['nose']['x'].index
            bins = [0, 18000, len(filled_keypoints_data['nose']['x']) - 18000, len(filled_keypoints_data['nose']['x'])]
            labels = [0, 2, 1]
            time_value = pd.cut(index, bins=bins, labels=labels, right=False, ordered=False)

            velocity_data = {}
            acceleration_data = {}
            avg_velocity_data = {}
            avg_acceleration_data = {}
            distance_traveled = {}
            for keypoint in keypoints:
                velocities, accelerations = calculate_keypoint(new_keypoints_data[m_id][keypoint])
                filter_velocities = velocities.where(velocities < 216)
                filled_velocities = fill_data(filter_velocities)
                velocity_data[keypoint] = filled_velocities
                acceleration_data[keypoint] = accelerations
                avg_velocity_data[keypoint] = pd.Series(velocity_data[keypoint]).rolling(window=11, center=True).mean()
                avg_acceleration_data[keypoint] = pd.Series(acceleration_data[keypoint]).rolling(window=11, center=True).mean()
            avg_velocity_data['all'] = np.mean(velocity_data['waist'])
            avg_acceleration_data['all'] = np.mean(acceleration_data['waist'])
            avg_velocity_data['baseline'] = velocity_data['waist'][time_value==0]
            avg_velocity_data['withrat'] = velocity_data['waist'][time_value==1]
            cv_velocity_data = {}
            cv_velocity_data['all'] = np.std(velocity_data['waist']) / np.mean(velocity_data['waist'])
            cv_velocity_data['baseline'] = np.std(velocity_data['waist'][time_value==0]) / np.mean(velocity_data['waist'][time_value==0])
            cv_velocity_data['withrat'] = np.std(velocity_data['waist'][time_value==1]) / np.mean(velocity_data['waist'][time_value==1])

            waist_data[m_id] = {'x': filled_keypoints_data['waist']['x'], 
                                'y': filled_keypoints_data['waist']['y']}
            waist_data_0[m_id] = {'x': filled_keypoints_data['waist']['x'][time_value == 0], 
                           'y': filled_keypoints_data['waist']['y'][time_value == 0]}
            waist_data_1[m_id] = {'x': filled_keypoints_data['waist']['x'][time_value == 1], 
                           'y': filled_keypoints_data['waist']['y'][time_value == 1]}
            distance_traveled = sum_distance_traveled(filled_keypoints_data['waist'])
            centroid_data_0 = {'x': filled_keypoints_data['centroid']['x'][time_value == 0], 
                               'y': filled_keypoints_data['centroid']['y'][time_value == 0]}
            
            location4lst = get_location_value(filled_keypoints_data['centroid'])
            location4lstbl = get_location_value_bl(filled_keypoints_data['centroid'])
            # print(location4ret.value_counts().get(0, 0), len(location4ret))
            zone_framecount = {'trigger_zone': location4lst.value_counts().get(3, 0),
                               'light_zone': location4lst.value_counts().get(2, 0),
                               'dark_zone': location4lst.value_counts().get(1, 0),
                               'nest_zone': location4lst.value_counts().get(0, 0)}
            
            immobile_framecount = {'all': np.sum(np.array(velocity_data['waist']) <= 1),
                                   'baseline': np.sum(np.array(velocity_data['waist'][time_value == 0]) <= 1),
                                   'withrat': np.sum(np.array(velocity_data['waist'][time_value == 1]) <= 1)}
            
            # head_angle = calculate_angle(nose_data_1[m_id], neck_data_1[m_id], 1100, method='ppl')
            # head_angle_framecount = {'le90': len(head_angle[head_angle <= 90]),
            #                          'gt90': len(head_angle[head_angle > 90])}
            
            trunk_keypoints = [filled_keypoints_data['nose'], 
                               filled_keypoints_data['neck'], 
                               filled_keypoints_data['waist'],
                               filled_keypoints_data['tail1']]
            trunk_length = calculate_distance_sum(trunk_keypoints)

            feature_data[m_id] = {
                'coordinate': filled_keypoints_data,
                'velocity': velocity_data,
                'acceleration': acceleration_data,
                'avg_velocity': avg_velocity_data,
                'cv_velocity':cv_velocity_data,
                'avg_acceleration': avg_acceleration_data,
                'time_value': time_value,
                'distance_traveled': distance_traveled,
                'location4lst': location4lst,
                'location4lstbl': location4lstbl, 
                'zone_framecount': zone_framecount,
                'immobile_framecount': immobile_framecount,
                # 'head_angle': head_angle,
                'trunk_length': trunk_length
                }

        distance_between = {'all': calculate_distance(waist_data[m_id_1], waist_data[m_id_2], method='pp'),
                            'baseline': calculate_distance(waist_data_0[m_id_1], waist_data_0[m_id_2], method='pp'),
                            'withrat': calculate_distance(waist_data_1[m_id_1], waist_data_1[m_id_2], method='pp')}
        
        feature_data[f'{m_id_1}_{m_id_2}'] = {'distance_between': distance_between}
    lstp_data = feature_data
    return lstp_data

def lst_read_decision_csv(lst_decision_file):
    trial_data = pd.read_csv(lst_decision_file, header=None, nrows=340)
    t_ids_dict = {'s':{'D':{}, 'S':{}}, 'p':{'D':{}, 'S':{}}}
    behaviors_dict = {'s':{'D':{}, 'S':{}}, 'p':{'D':{}, 'S':{}}}
    freeze_time_dict = {'s':{'D':{}, 'S':{}}, 'p':{'D':{}, 'S':{}}}
    for i in range(len(trial_data[0])):
        t_id = trial_data[0][i]
        video_file = trial_data[1][i]
        looming_start_frame = trial_data[3][i]
        m1_behavior = trial_data[4][i]
        m2_behavior = trial_data[5][i]
        m1_trigger_value = trial_data[6][i]
        m2_trigger_value = trial_data[7][i]
        # m1_freeze_start_frame = trial_data[8][i]

        if m1_trigger_value == 0:
            m1_behavior = 'n'
        if m2_trigger_value == 0:
            m2_behavior = 'n'
        group_num = t_id.split('B')[1].split('M')[0].split('D')[0]
        if group_num in ['1', '3', '7', '10', '11', '12', '13', '14', '15', '16', '17', '20']:
            if 'M1' in t_id:
                ts_id = t_id.split('M')[0] + 'MST'+ t_id.split('T')[1]
                behaviors_dict['s']['S'][ts_id] = m1_behavior
            elif 'M2' in t_id:
                td_id = t_id.split('M')[0] + 'MDT'+ t_id.split('T')[1]
                behaviors_dict['s']['D'][td_id] = m2_behavior
            else:
                td_id = t_id.split('D')[0] + 'MDT'+ t_id.split('T')[1]
                behaviors_dict['p']['D'][td_id] = m2_behavior
                ts_id = t_id.split('D')[0] + 'MST'+ t_id.split('T')[1]
                behaviors_dict['p']['S'][ts_id] = m1_behavior
        else:
            if 'M1' in t_id:
                td_id = t_id.split('M')[0] + 'MDT' + t_id.split('T')[1]
                behaviors_dict['s']['D'][td_id] = m1_behavior
            elif 'M2' in t_id:
                ts_id = t_id.split('M')[0] + 'MST'+ t_id.split('T')[1]
                behaviors_dict['s']['S'][ts_id] = m2_behavior
            else:
                td_id = t_id.split('D')[0] + 'MDT'+ t_id.split('T')[1]
                behaviors_dict['p']['D'][td_id] = m1_behavior
                ts_id = t_id.split('D')[0] + 'MST'+ t_id.split('T')[1]
                behaviors_dict['p']['S'][ts_id] = m2_behavior
    # print(behaviors_dict['s']['D'])
    # print(freeze_time_dict)
    return behaviors_dict

def ret_reorganize_frames_data(behavior_frames_labels_dict):
    reorganized_data = {'s': {'D': {}, 'S': {}}, 'p': {'D': {}, 'S': {}}}
    for key, value in behavior_frames_labels_dict.items():
        if len(key) == 4 or len(key) == 5:
            if key[1] in ['4']:
                rank = 'S' if key[-1] == '1' else 'D'
                t_id = key[:-2] + 'M' + rank
            else:
                rank = 'D' if key[-1] == '1' else 'S'
                t_id = key[:-2] + 'M' + rank

            for sub_key, sub_value in value.items():
                if t_id not in reorganized_data['s'][rank]:
                    reorganized_data['s'][rank][t_id] = {}
                reorganized_data['s'][rank][t_id][sub_key] = sub_value
        elif len(key) == 7 or len(key) == 8:
            td_id = key[:-5] + 'MD'
            ts_id = key[:-5] + 'MS'
            for sub_key, sub_value in value.items():
                if sub_key.startswith('MD_'):
                    if td_id not in reorganized_data['p']['D']:
                        reorganized_data['p']['D'][td_id] = {}
                    reorganized_data['p']['D'][td_id][sub_key[3:]] = sub_value
                elif sub_key.startswith('MS_'):
                    if ts_id not in reorganized_data['p']['S']:
                        reorganized_data['p']['S'][ts_id] = {}
                    reorganized_data['p']['S'][ts_id][sub_key[3:]] = sub_value

    return reorganized_data

def ret_read_deepethogram_csv(ret_labels_dir, bento_correction_file):
    behavior_frames_labels_dict = {}
    for file_name in os.listdir(ret_labels_dir):
        if file_name.endswith('.csv'):
            key = file_name.split('_')[-2]
            behavior_frames_labels_dict[key] = {}
            if file_name.split('M')[0] == 'D12':
                continue
            if len(key) in [4, 5, 7, 8]:
                file_path = os.path.join(ret_labels_dir, file_name)
                data = pd.read_csv(file_path)
                data = data.drop(data.columns[[0, 1]], axis=1)
                # data.columns = [header[3:] for header in data.columns]
                content = data.values
                for header in data.columns:
                    behavior_frames_labels_dict[key][header] = data[header]

    origin_behavior_frames_labels_dict = copy.deepcopy(behavior_frames_labels_dict)
    correction_data = pd.read_csv(bento_correction_file, header=None, usecols=[0, 1])  # real withrat start frame
    correction_dict = dict(zip(correction_data[0], correction_data[1]))
    for key in behavior_frames_labels_dict.keys():  # correct behavior_frames_labels_dict
        if key in correction_dict:
            correction_value = correction_dict[key]
            for sub_key in behavior_frames_labels_dict[key]:
                original_series = behavior_frames_labels_dict[key][sub_key]
                corrected_series = original_series.shift(-correction_value)
                corrected_series = corrected_series[corrected_series.index >= 0]
                behavior_frames_labels_dict[key][sub_key] = corrected_series
    origin_behavior_frames_labels_dict = ret_reorganize_frames_data(origin_behavior_frames_labels_dict)
    behavior_frames_labels_dict = ret_reorganize_frames_data(behavior_frames_labels_dict)

    return behavior_frames_labels_dict, origin_behavior_frames_labels_dict


def ret_read_annotation_file(file_path):
    behavior_frames = {}
    behaviors = [
        'rat_in',
        'approach',
        'investigation',
        'withdrawal',
        'stretch-attend',
        'freezing',
        'tail_rattling',
        'huddling',
        'approach_partner',
        'follow_partner',
        'groom_partner',
        'sniff_partner',
        'grooming']
    # initialize behavior_frames with all possible behaviors
    for behavior in behaviors:
        behavior_frames[behavior] = []
    
    # Extract the file name from the path (without the extension)
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    
    if len(file_name) == 4 or len(file_name) == 5:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        current_behavior = None
        for line in lines:
            if line.startswith('>'):
                current_behavior = line.strip()[1:]
                behavior_frames[current_behavior] = []
            elif line.strip() != '':
                try:
                    start_frame, stop_frame, _ = map(int, line.split())
                    behavior_frames[current_behavior].append((start_frame, stop_frame))
                except ValueError:
                    pass
    elif len(file_name) == 7 or len(file_name) == 8:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        chd_dict = {}
        chs_dict = {}
        current_dict = {}
        current_behavior = None
        for behavior in behaviors:
            chd_dict[behavior] = []
            chs_dict[behavior] = []
        for line in lines:
            if line.startswith('Channel_D----------'):
                current_dict = chd_dict
            elif line.startswith('Channel_S----------'):
                current_dict = chs_dict
            elif line.strip() != '':
                if line.startswith('>'):
                    current_behavior = line.strip()[1:]
                    current_dict[current_behavior] = []
                else:
                    try:
                        start_frame, stop_frame, _ = map(int, line.split())
                        current_dict[current_behavior].append((start_frame, stop_frame))
                    except ValueError:
                        pass
        behavior_frames = {'ChD': chd_dict, 'ChS': chs_dict}
    return behavior_frames


def ret_correct_bento_data(behavior_frames_dict, correction_data):
    corrected_dict = behavior_frames_dict.copy()
    for key, value in behavior_frames_dict.items():
        for sub_key, sub_value in value.items():
            if isinstance(sub_value, list):
                correction_row = correction_data[correction_data[0] == key]
                correction_value = correction_row.iloc[0, 1]
                corrected_frames = []
                for start_frame, stop_frame in sub_value:
                    corrected_frames.append((start_frame - correction_value, stop_frame - correction_value))
                corrected_dict[key][sub_key] = corrected_frames
            else:
                for sub_sub_key, sub_sub_value in sub_value.items():
                    correction_row = correction_data[correction_data[0] == key]
                    correction_value = correction_row.iloc[0, 1]
                    corrected_frames = []
                    for start_frame, stop_frame in sub_sub_value:
                        corrected_frames.append((start_frame - correction_value, stop_frame - correction_value))
                    corrected_dict[key][sub_key][sub_sub_key] = corrected_frames
    return corrected_dict


def ret_reorganize_labels_data(behavior_frames_dict):
    reorganized_data = {'s': {'D': {}, 'S': {}}, 'p': {'D': {}, 'S': {}}}
    for key, value in behavior_frames_dict.items():
        if len(key) == 4 or len(key) == 5:
            if key[1] in ['4']:
                if key[-1] == '1':
                    rank = 'S'
                elif key[-1] == '2':
                    rank = 'D'
                t_id = key[:-2] + 'M' + rank
                reorganized_data['s'][rank][t_id] = value
            else:
                if key[-1] == '1':
                    rank = 'D'
                elif key[-1] == '2':
                    rank = 'S'
                t_id = key[:-2] + 'M' + rank
                reorganized_data['s'][rank][t_id] = value
        elif len(key) == 7 or len(key) == 8:
            td_id = key[:-5] + 'MD'
            ts_id = key[:-5] + 'MS'
            for sub_key, sub_value in value.items():
                if sub_key == 'ChD':
                    reorganized_data['p']['D'][td_id] = sub_value
                elif sub_key == 'ChS':
                    reorganized_data['p']['S'][ts_id] = sub_value
    return reorganized_data


def ret_read_bento_annot(bento_path, bento_correction_file):
    behavior_frames_dict = {}  # all bento data
    for file_name in os.listdir(bento_path):
        if file_name.endswith('.annot'):
            t_id = file_name[:-6]
            if t_id.split('M')[0] == 'D12':
                continue
            file_path = os.path.join(bento_path, file_name)
            behavior_frames = ret_read_annotation_file(file_path)
            behavior_frames_dict[t_id] = behavior_frames

    # print(behavior_frames_dict.keys())
    correction_data = pd.read_csv(bento_correction_file, header=None, usecols=[0, 1])  # real withrat start frame
    origin_behavior_frames_dict = copy.deepcopy(behavior_frames_dict)
    corrected_behavior_frames_dict = ret_correct_bento_data(behavior_frames_dict, correction_data)  # correct frame

    behavior_frames_dict_p = {}  # pair bento data, {'BXDXTX': {'ChD: sub_dict, 'ChS': sub_dict}}
    behavior_frames_dict_s = {}  # single bento data, {'BXMXDXTX': sub_dict}
    for key, value in behavior_frames_dict.items():
        if len(key) in [7, 8]:
            behavior_frames_dict_p[key] = value
        else:
            behavior_frames_dict_s[key] = value
    behavior_frames_dict_p_new = {}  # this dict is the same as the structure of behavior_frames_dict_s
    for key, value in behavior_frames_dict_p.items():
        if key.split('D')[1].split('M')[0].split('M')[0] == '5':
            new_key_1 = key.split('M')[0].split('M')[0]+'M2'
            new_key_2 = key.split('M')[0].split('M')[0]+'M1'
        else:
            new_key_1 = key.split('M')[0].split('M')[0]+'M1'
            new_key_2 = key.split('M')[0].split('M')[0]+'M2'
        
        behavior_frames_dict_p_new[new_key_1] = value['ChD']
        behavior_frames_dict_p_new[new_key_2] = value['ChS']

    origin_behavior_frames_dict = ret_reorganize_labels_data(origin_behavior_frames_dict)
    behavior_frames_dict = ret_reorganize_labels_data(behavior_frames_dict)
    return behavior_frames_dict, origin_behavior_frames_dict


def ret_rename_keys_MX2MR(data, trans_rank_values):
    renamed_data = {}
    for category, keys in data.items():
        renamed_data[category] = {}
        for key, sub_keys in keys.items():
            if key != 'Thi':
                no_string_key = key.replace('_', '')
                renamed_data[category][no_string_key] = {}
                if category == 'single':
                    trans_rank_value = trans_rank_values[category][key]
                    for sub_key, value in sub_keys.items():
                        if trans_rank_value == 0:
                            new_sub_key = 'MD' if sub_key == 'M1' else 'MS'
                        else:
                            new_sub_key = 'MS' if sub_key == 'M1' else 'MD'
                        if new_sub_key not in renamed_data[category][no_string_key]:
                            renamed_data[category][no_string_key][new_sub_key] = {}
                        renamed_data[category][no_string_key][new_sub_key] = value
                elif category == 'pair':
                    for sub_key1, sub_keys2 in sub_keys.items():
                        for sub_key2, value in sub_keys2.items():
                            trans_rank_value = trans_rank_values[category][sub_key2][key]
                            if trans_rank_value == 0:
                                new_sub_key1 = 'MD' if sub_key1 == 'M1' else 'MS'
                            else:
                                new_sub_key1 = 'MS' if sub_key1 == 'M1' else 'MD'
                            if new_sub_key1 not in renamed_data[category][no_string_key]:
                                renamed_data[category][no_string_key][new_sub_key1] = {}
                            renamed_data[category][no_string_key][new_sub_key1][sub_key2] = value
    return renamed_data


def reorder_dict_sessions(data_dict, session_order):
    from collections import OrderedDict
    reordered_dict = {}
    for group in data_dict:
        reordered_dict[group] = {}
        for rank in data_dict[group]:
            reordered_dict[group][rank] = OrderedDict()
            for session_id in session_order:
                if session_id in data_dict[group][rank]:
                    reordered_dict[group][rank][session_id] = data_dict[group][rank][session_id]
            for session_id in data_dict[group][rank]:
                if session_id not in session_order:
                    reordered_dict[group][rank][session_id] = data_dict[group][rank][session_id]
    
    return reordered_dict


def ret_read_ethovision_xlsx_and_analyze_data(ev_file_path):
    raw_data = {}
    for file in os.listdir(ev_file_path):
        if file.startswith('Raw data') and file.endswith('.xlsx'):
            xlsx_file_path = os.path.join(ev_file_path, file)
            if 'Thigmotaxis' not in file:
                if "pair" in file.lower():
                    category = "pair"
                elif "single" in file.lower():
                    category = "single"
            else:
                continue
            key = file.split('-')[1].split('_')[0]
            if category == "pair":
                sub_key2 = "baseline" if "baseline" in file else "withrat"
                df1 = pd.read_excel(xlsx_file_path, sheet_name=0, skiprows=34)
                df2 = pd.read_excel(xlsx_file_path, sheet_name=1, skiprows=34)
                df1 = df1.drop(df1.index[0]).reset_index(drop=True)
                df2 = df2.drop(df2.index[0]).reset_index(drop=True)
                trial_start_time = df1.iloc[0, 0]
                trial_start_frame = int(trial_start_time * 30)
                num_columns = df1.shape[1]
                empty_rows = pd.DataFrame(np.nan, index=range(trial_start_frame), columns=df1.columns)
                df1 = pd.concat([empty_rows, df1]).reset_index(drop=True)
                df2 = pd.concat([empty_rows, df2]).reset_index(drop=True)
                raw_data_dict1 = {col: df1[col] for col in df1.columns[1:]}
                raw_data_dict2 = {col: df2[col] for col in df2.columns[1:]}
                if category not in raw_data:
                    raw_data[category] = {}
                if key not in raw_data[category]:
                    raw_data[category][key] = {}
                    raw_data[category][key]['M1'] = {}
                    raw_data[category][key]['M2'] = {}
                if sub_key2 not in raw_data[category][key]['M1']:
                    raw_data[category][key]['M1'][sub_key2] = {}
                if sub_key2 not in raw_data[category][key]['M2']:
                    raw_data[category][key]['M2'][sub_key2] = {}
                raw_data[category][key]['M1'][sub_key2] = raw_data_dict1
                raw_data[category][key]['M2'][sub_key2] = raw_data_dict2
            elif category == "single":
                sub_key1 = "M1" if "M1" in file else "M2"
                sub_key2 = "baseline" if "baseline" in file else "withrat"
                df = pd.read_excel(xlsx_file_path, skiprows=34)
                df = df.drop(df.index[0]).reset_index(drop=True)
                raw_data_dict = {col: df[col] for col in df.columns[1:]}
                if category not in raw_data:
                    raw_data[category] = {}
                if key not in raw_data[category]:
                    raw_data[category][key] = {}
                if sub_key1 not in raw_data[category][key]:
                    raw_data[category][key][sub_key1] = {}
                if sub_key2 not in raw_data[category][key][sub_key1]:
                    raw_data[category][key][sub_key1][sub_key2] = {}
                raw_data[category][key][sub_key1][sub_key2] = raw_data_dict
    print("all raw data in ethovision xlsx are read.")

    stat_data = {}
    for file in os.listdir(ev_file_path):
        if file.startswith('Statistics') and file.endswith('.xlsx'):
            xlsx_file_path = os.path.join(ev_file_path, file)
            if "pair" in file.lower():
                category = "pair"
            elif "single" in file.lower():
                category = "single"
            key = file.split('-')[1].split('_')[0]
            if category == "pair":
                sub_key2 = "baseline" if "baseline" in file else "withrat"
                df = pd.read_excel(xlsx_file_path, sheet_name=0)
                df = df.drop(df.columns[:2], axis=1)
                df1 = df.iloc[0]
                df2 = df.iloc[1]
                df1 = df1.to_frame().T
                df2 = df2.to_frame().T
                raw_data_dict1 = df1.to_dict(orient='records')
                raw_data_dict2 = df2.to_dict(orient='records')
                raw_data_dict1 = raw_data_dict1[0]
                raw_data_dict2 = raw_data_dict2[0]
                if category not in stat_data:
                    stat_data[category] = {}
                if key not in stat_data[category]:
                    stat_data[category][key] = {}
                    stat_data[category][key]['M1'] = {}
                    stat_data[category][key]['M2'] = {}
                stat_data[category][key]['M1'][sub_key2] = raw_data_dict1
                stat_data[category][key]['M2'][sub_key2] = raw_data_dict2
            elif category == "single":
                sub_key1 = "M1" if "M1" in file else "M2"
                sub_key2 = "baseline" if "baseline" in file else "withrat"
                sub_key2 = "baseline" if "baseline" in file else "withrat"
                df = pd.read_excel(xlsx_file_path, sheet_name=0)
                df = df.drop(df.columns[0], axis=1)
                raw_data_dict = df.to_dict(orient='records')
                raw_data_dict = raw_data_dict[0]
                if category not in stat_data:
                    stat_data[category] = {}
                if key not in stat_data[category]:
                    stat_data[category][key] = {}
                if sub_key1 not in stat_data[category][key]:
                    stat_data[category][key][sub_key1] = {}
                stat_data[category][key][sub_key1][sub_key2] = raw_data_dict
    print("all statistics data in ethovision xlsx are read.")
    trans_rank_values = {
        'single': {'D1': 0, 'D2': 0, 'D3': 0, 'D4': 1, 'D5': 0, 'D6': 0, 
                   'D7': 0, 'D8': 0, 'D9': 0, 'D10': 0, 'D11': 0, 'D12': 0},
        'pair': {'baseline': {'D1': 0, 'D2': 1, 'D3': 0, 'D4': 1, 'D5': 0, 'D6': 0, 
                              'D7': 1, 'D8': 0, 'D9': 0, 'D10': 0, 'D11': 1, 'D12': 1}, 
                 'withrat': {'D1': 0, 'D2': 1, 'D3': 0, 'D4': 1, 'D5': 0, 'D6': 0, 
                             'D7': 0, 'D8': 0, 'D9': 0, 'D10': 0, 'D11': 0, 'D12': 1}}}
    renamed_raw_data = ret_rename_keys_MX2MR(raw_data, trans_rank_values)
    renamed_stat_data = ret_rename_keys_MX2MR(stat_data, trans_rank_values)
    return renamed_raw_data, renamed_stat_data


def ret_extract_target_data(data, target_key, categories=['single', 'pair'], groups=['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11'], ranks=['MD', 'MS'], times=['baseline', 'withrat']):
    extracted_data = {}
    for category in categories:
        for rank in ranks:
            for time in times:
                for group in groups:
                    if time in data[category][group][rank].keys():
                        if category not in extracted_data.keys():
                            extracted_data[category] = {}
                        if rank not in extracted_data[category].keys():
                            extracted_data[category][rank] = {}
                        if time not in extracted_data[category][rank].keys():
                            extracted_data[category][rank][time] = {}
                        if target_key[0] in data[category][group][rank][time].keys():
                            target_value = data[category][group][rank][time][target_key[0]]
                        elif len(target_key) >= 2 and target_key[1] in data[category][group][rank][time].keys():
                            target_value = data[category][group][rank][time][target_key[1]]
                        else:
                            continue
                        extracted_data[category][rank][time][group] = target_value
    return extracted_data


def get_avg_dict2list(dict_data, categories, sub_categories, nan2zero=False):
    result_data = {}
    for category in categories:
        result_data[category] = {}
        for sub_category in sub_categories:
            result_data[category][sub_category] = {}
            index_group_values = {}
            for key, value in dict_data[category][sub_category].items():
                if nan2zero == True:
                    if np.isnan(value).any():
                        value = [0]  # replace NaN with 0
                index_group = key.split('T')[0]
                if index_group not in index_group_values:
                    index_group_values[index_group] = []
                index_group_values[index_group].append([value[0]])
            
            for index_group, values in index_group_values.items():
                avg_value = np.nanmean(values)
                result_data[category][sub_category][index_group] = avg_value
    return result_data


def lst_mean_mouse_in_dict(dict, nan2zero=False):
    result = {}
    sD_dict = get_avg_dict2list(dict, categories=['s'], sub_categories=['D'], nan2zero=nan2zero)
    sS_dict = get_avg_dict2list(dict, categories=['s'], sub_categories=['S'], nan2zero=nan2zero)
    pD_dict = get_avg_dict2list(dict, categories=['p'], sub_categories=['D'], nan2zero=nan2zero)
    pS_dict = get_avg_dict2list(dict, categories=['p'], sub_categories=['S'], nan2zero=nan2zero)
    result['s'] = {'D': sD_dict['s']['D'], 'S': sS_dict['s']['S']}
    result['p'] = {'D': pD_dict['p']['D'], 'S': pS_dict['p']['S']}
    return result


def dict_to_dataframe(dict_data, value_name='', groups=['s', 'p'], ranks=['D', 'S'], nan2zero=False, method='mean'):
    """
    Convert a nested dictionary {group: {rank: {trial_id: value}}} to a DataFrame.

    Parameters:
    -----------
    dict_data : dict
        Nested dictionary with the format {group: {rank: {trial_id: value}}}
    value_name : str
        Column name prefix for the values, e.g. 'defense_pct', 'first_freezing_duration'
    groups : list
        List of groups to process; default ['s', 'p']
    ranks : list
        List of ranks to process; default ['D', 'S']
    method : str
        Aggregation method for multiple trials from the same mouse: 'mean' (default) or 'sum'

    Returns:
    --------
    pd.DataFrame
        The converted DataFrame, where each row represents a mouse (B ID) and columns contain B_id and condition-specific values (e.g. value_sD, value_sS, value_pD, value_pS).
        Multiple trials from the same mouse are aggregated (by mean or sum, depending on method).
    """
    import pandas as pd
    import numpy as np
    
    # Create a dictionary to store values for each B ID under different conditions
    # {B_id: {col_name: [values]}}
    b_data = {}
    
    # Iterate over each group
    for g in groups:
        if g not in dict_data:
            continue

        # Iterate over each rank
        for rank in ranks:
            if rank not in dict_data[g]:
                continue

            # Iterate over each trial_id
            for trial_id, value in dict_data[g][rank].items():
                # Extract the B ID (e.g. 'B1MDT1' -> 'B1')
                b_id = trial_id.split('M')[0]

                # Initialize the entry if B_id does not exist yet
                if b_id not in b_data:
                    b_data[b_id] = {}

                # Create the column name: {value_name}_{group}{rank}
                col_name = f"{value_name}_{g}{rank}"

                # Handle different types of values
                if isinstance(value, list):
                    if len(value) == 1:
                        processed_value = value[0]  # A list with a single element: take the first one
                    else:
                        processed_value = np.nanmean(value)  # Multiple elements: calculate the mean
                else:
                    processed_value = value

                # Collect all values for this condition (for later averaging)
                if col_name not in b_data[b_id]:
                    b_data[b_id][col_name] = []
                b_data[b_id][col_name].append(processed_value)

    # Average each condition for each B
    data_dict = {}
    for b_id, cols in b_data.items():
        data_dict[b_id] = {'B_id': b_id}
        for col_name, values in cols.items():
            if nan2zero:
                values = [0 if np.isnan(v) else v for v in values]
            if method == 'sum':
                data_dict[b_id][col_name] = np.nansum(values)
            else:
                data_dict[b_id][col_name] = np.nanmean(values)
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(data_dict, orient='index')
    
    if len(df) == 0:
        return df
    
    # Sort by B ID
    df['B_num'] = df['B_id'].str.extract('(\d+)').astype(int)
    df = df.sort_values('B_num').reset_index(drop=True)
    df.drop(columns=['B_num'], inplace=True)  # Remove the helper column
    
    # Reorder columns in the order B_id, sD, sS, pD, pS
    ordered_cols = ['B_id']
    for g in groups:
        for rank in ranks:
            col_name = f"{value_name}_{g}{rank}"
            if col_name in df.columns:
                ordered_cols.append(col_name)
    
    # Keep only the existing columns to avoid KeyError
    ordered_cols = [col for col in ordered_cols if col in df.columns]
    df = df[ordered_cols]
    
    return df


def filter_dict_data(data, behavior):
    filtered_data = {}
    for key, value in data.items():
        if isinstance(value, dict):
            data = filter_dict_data(value, behavior)
            if data:  
                filtered_data[key] = data
        elif key == behavior:
            if value: 
                filtered_data = value
            else:
                filtered_data = [np.nan]
    return filtered_data


def filter_in_range(data, t_range, method='delete'):
    if isinstance(data, dict):
        filtered_dict = {k: v for k, v in ((k, filter_in_range(v, t_range, method)) for k, v in data.items()) if v is not None}
        return filtered_dict if filtered_dict else np.nan
    elif isinstance(data, list):
        filtered_list = [t for t in (filter_in_range(t, t_range, method) for t in data) if t is not None]
        return filtered_list if filtered_list else [np.nan]
    elif isinstance(data, tuple):
        start, end = data
        if method == 'replace':
            if end < t_range[0] or start > t_range[1]:
                return None
            start = max(start, t_range[0])
            end = min(end, t_range[1])
            return (start, end)
        elif method == 'delete':
            if start >= t_range[0] and end <= t_range[1]:
                return (start, end)
            return None
    return data


def calculate_keypoint(kp):
    dx = kp['x'].diff(periods=1)
    dy = kp['y'].diff(periods=1)

    indices = kp['x'].index.values[0:]
    velocities = np.sqrt(dx ** 2 + dy ** 2)
    velocities = pd.Series(velocities, index=indices)
    indices = kp['x'].index.values[0:]
    accelerations = np.diff(velocities)
    accelerations = np.concatenate(([np.nan], accelerations))
    accelerations = pd.Series(accelerations, index=indices)
    return velocities, accelerations


def calculate_angle(kp1, kp2, kp3, method):
    if method == 'ppp': # calculate point1-point2-point3 angle
        df1 = pd.concat(kp1, axis=1)
        df2 = pd.concat(kp2, axis=1)
        df3 = pd.concat(kp3, axis=1)
    
        merged_data = pd.merge(df1, df2, left_index=True, right_index=True)
        merged_data = pd.merge(merged_data, df3, left_index=True, right_index=True)
    
        v1_x = merged_data.iloc[:, 0] - merged_data.iloc[:, 2]
        v1_y = merged_data.iloc[:, 1] - merged_data.iloc[:, 3]
        v1 = pd.DataFrame({'v1_x': v1_x, 'v1_y': v1_y})
    
        v3_x = merged_data.iloc[:, 4] - merged_data.iloc[:, 2]
        v3_y = merged_data.iloc[:, 5] - merged_data.iloc[:, 3]
        v3 = pd.DataFrame({'v3_x': v3_x, 'v3_y': v3_y})
    
        angles = pd.Series(index=v1.index)
        for i in range(len(v1)):
            dot_product = v1.iloc[i, 0] * v3.iloc[i, 0] + v1.iloc[i, 1] * v3.iloc[i, 1]
            norm_v1 = np.linalg.norm(v1.iloc[i, :])
            norm_v3 = np.linalg.norm(v3.iloc[i, :])
            if norm_v1 == 0 or norm_v3 == 0:
                angles[i] = None
            else:
                angle_rad = np.arccos(dot_product / (norm_v1 * norm_v3))
                angle_deg = np.degrees(angle_rad)
                angles[i] = angle_deg
    elif method == 'ppl': # calculate point1-point2-foot_point angle
        df1 = pd.concat(kp1, axis=1).reset_index(drop=True)
        df2 = pd.concat(kp2, axis=1).reset_index(drop=True)
    
        v_x = df2.iloc[:, 0] - df1.iloc[:, 0]
        v_y = df2.iloc[:, 1] - df1.iloc[:, 1]
        angles = pd.Series(index=v_x.index)
        for i in range(len(v_x)):
            norm_v = np.linalg.norm([v_x.iloc[i], v_y.iloc[i]])
            if norm_v == 0:
                angles[i] = None
            else:
                angle_rad = np.arccos(v_x.iloc[i] / norm_v)
                angle_deg = np.degrees(angle_rad)
                angles[i] = angle_deg
    else:
        print('Invalid method. Choose either "ppl" or "ppp".')
        return None
    angles = angles.reset_index(drop=True)
    return angles


def sum_distance_traveled(kp):
    x_diff = np.diff(np.nan_to_num(kp['x']))
    y_diff = np.diff(np.nan_to_num(kp['y']))
    trajectory_length = np.sqrt(x_diff**2 + y_diff**2)
    trajectory_length_list = []
    for i in range(len(kp['x'])):
        trajectory_length_i = np.sum(trajectory_length[:i])
        trajectory_length_list.append(trajectory_length_i)
    return pd.Series(trajectory_length_list)


def calculate_distance(kp1, kp2, method):
    if method == 'pl': # calculate point-to-line distance
        distance = kp1['x'] - kp2
    elif method == 'pp': # calculate point-to-point distance
        distance = np.sqrt((kp1['x'] - kp2['x']) ** 2 + (kp1['y'] - kp2['y']) ** 2)
    else:
        print('Invalid method. Choose either "pl" or "pp".')
        return None
    return distance


def calculate_distance_sum(kps):
    distance_sum = calculate_distance(kps[0], kps[1], method='pp') + \
                   calculate_distance(kps[1], kps[2], method='pp') + \
                   calculate_distance(kps[2], kps[3], method='pp')
    return distance_sum


def get_lst_location_value(kp, edge_width=43.52*4):
    """0=nest, 1=edge, 2=center"""
    pixpergrid = 43.52
    arena_size = 1088

    nest_crood = [0, pixpergrid * 17.5]
    nest_size = [pixpergrid * 10, pixpergrid * 7.5]

    location_value = pd.Series(np.full_like(kp['x'], 2))
    edge_mask = (
        (kp['x'] <= edge_width) | (kp['x'] >= arena_size - edge_width) |
        (kp['y'] <= edge_width) | (kp['y'] >= arena_size - edge_width)
    )
    location_value[edge_mask] = 1
    nest_mask = (
        (kp['x'] >= nest_crood[0]) & (kp['x'] <= nest_crood[0] + nest_size[0]) &
        (kp['y'] >= nest_crood[1]) & (kp['y'] <= nest_crood[1] + nest_size[1])
    )
    location_value[nest_mask] = 0
    return location_value


def get_ret_location_value(x_values):
    """Assign region labels: 0=near (close to rat), 1=middle, 2=far"""
    
    dx = (19 + 5) / 24  # cm per unit (= 1.0)
    rat_x = -5           # X coordinate start on the rat side (cm)
    arena_size = 24 * dx  # Total arena length: 24 cm

    # Region definition: near:middle:far = 9:9:6 (cm)
    near_zone_width   = 9 * dx
    middle_zone_width = 9 * dx
    far_zone_width    = 6 * dx

    near_zone_range   = (rat_x, rat_x + near_zone_width)                                           # (-5,  4)
    middle_zone_range = (rat_x + near_zone_width, rat_x + near_zone_width + middle_zone_width)     # ( 4, 13)
    far_zone_range    = (rat_x + near_zone_width + middle_zone_width, rat_x + arena_size)          # (13, 19)

    x = x_values.reset_index(drop=True)
    loc = pd.Series(np.full(len(x), np.nan))
    near_mask = (x >= near_zone_range[0]) & (x < near_zone_range[1])
    # near_mask = (x <= near_zone_range[1])
    middle_mask = (x >= middle_zone_range[0]) & (x < middle_zone_range[1])
    far_mask = (x >= far_zone_range[0]) & (x <= far_zone_range[1])
    # far_mask = (x >= far_zone_range[0])
    loc[near_mask.values]   = 0
    loc[middle_mask.values] = 1
    loc[far_mask.values]    = 2
    return loc


def calculate_latency_time(data, fps=30):
    """
    Calculate latency time (time from looming onset to behavior onset).

    Parameters:
    -----------
    data : dict
        Nested dictionary whose innermost values are behavior interval lists [(start, end), ...]
    fps : int
        Frame rate; default is 30

    Returns:
    --------
    dict
        A dictionary containing latency times with the same structure as the input.
    """
    latency_data = {}
    for key, value in data.items():
        if isinstance(value, dict):
            # Recursively process nested dictionaries
            latency_data[key] = calculate_latency_time(value, fps)
        elif isinstance(value, list):
            # Process the interval list
            latencies = []
            for item in value:
                if isinstance(item, tuple) and len(item) == 2:
                    start, end = item
                    # Compute the latency from frame 150 (looming onset) to behavior onset
                    latency = (start - 150) / fps
                    latencies.append(latency)
                elif np.isnan(item).all() if hasattr(item, '__iter__') else np.isnan(item):
                    latencies.append(np.nan)
            
            # If there is no valid data, return [np.nan]
            if len(latencies) == 0:
                latencies = [np.nan]
            
            latency_data[key] = latencies
        else:
            # Handle a single value or NaN
            latency_data[key] = [np.nan]
    
    return latency_data


def filter_duration(data, min_duration=15):
    for key in data.keys():
        for sub_key in data[key].keys():
            for sub_key2 in data[key][sub_key].keys():
                tuples = data[key][sub_key][sub_key2]
                filtered_data = []
                if np.isnan(tuples).any():
                    continue
                for start, end in sorted(tuples):
                    if end-start+1>=min_duration:  # +1 for inclusive counting
                        filtered_data.append((start, end))
                if len(filtered_data) <= 0:
                    filtered_data.append(np.nan)
                data[key][sub_key][sub_key2] = filtered_data
    return data


def calculate_duration_time(data, first=False, max_drt=False, unit='seconds', framerate=30):
    duration_data = {}
    for key, values in data.items():
        if isinstance(values, dict):
            duration_data[key] = calculate_duration_time(values, first=first, max_drt=max_drt, unit=unit, framerate=framerate)
        else:
            if first:
                if len(values) != 0:
                    values = [values[0]]
            durations = []
            if np.isnan(values).any():
                duration = np.nan
                durations.append(duration)
                duration_data[key] = durations
            else:
                for value in values:
                    start, end = value
                    if unit == 'seconds' and framerate:
                        duration = (end - start +1) / framerate  # +1 for inclusive counting
                    elif unit == 'frames' and not framerate:
                        duration = (end - start +1)  # +1 for inclusive counting
                    durations.append(duration)
                if max_drt:
                    duration_data[key] = [max(durations)]
                else:
                    duration_data[key] = durations
    return duration_data


def calculate_bouts(data):
    count_data = {}
    for key, values in data.items():
        if isinstance(values, dict):
            count_data[key] = calculate_bouts(values)
        else:
            if np.all(np.isnan(values)):
                count_data[key] = [0]
            else:
                count_data[key] = [len(values)]
    return count_data


def calculate_average_behavior_durations(data):
    averages = {}
    for subject, trials in data.items():
        averages[subject] = {}
        for trial_type, trial_data in trials.items():
            averages[subject][trial_type] = {}
            for trial, durations in trial_data.items():
                avg_duration = np.nanmean(durations)
                averages[subject][trial_type][trial] = [avg_duration]
    return averages


def calculate_total(data):
    total_data = {}
    for key, value in data.items():
        if isinstance(value, dict):
            total_data[key] = calculate_total(value)
        elif key:
            total = sum(value)
            total_data[key] = [total]
    return total_data


def calculate_duration_pct(data, frame_range=None, time_range=None, framerate=30):
    # Calculate total duration (seconds)
    if frame_range is not None:
        total_time = (frame_range[1] - frame_range[0]) / framerate
    elif time_range is not None:
        total_time = time_range[1] - time_range[0]
    else:
        raise ValueError("Either frame_range or time_range must be provided")
    
    pct_data = {}
    for key, value in data.items():
        if isinstance(value, dict):
            # Recursively process nested dictionaries
            pct_data[key] = calculate_duration_pct(
                value, 
                frame_range=frame_range,
                time_range=time_range, 
                framerate=framerate
            )
        elif isinstance(value, list):
            # Process duration lists and convert them to percentages
            percentages = []
            for duration in value:
                if np.isnan(duration):
                    percentages.append(np.nan)
                else:
                    pct = (duration / total_time) * 100
                    percentages.append(pct)
            pct_data[key] = percentages
        else:
            # Process a single value
            if np.isnan(value):
                pct_data[key] = np.nan
            else:
                pct_data[key] = (value / total_time) * 100
    
    return pct_data


def merge_dicts(dict1, dict2):
    merged_dict = {'s': {'D': {}, 'S': {}}, 'p': {'D': {}, 'S': {}}}
    for dct in [dict1, dict2]:
        for main_key in ['s', 'p']:
            for sub_key, sub_value in dct[main_key].items():
                # Ensure sub_key exists in merged_dict
                if sub_key not in merged_dict[main_key]:
                    merged_dict[main_key][sub_key] = {}
                
                for sub_sub_key, sub_sub_value in dct[main_key][sub_key].items():
                    if sub_sub_key in merged_dict[main_key][sub_key]:
                        # The key already exists and needs to be merged
                        existing_value = merged_dict[main_key][sub_key][sub_sub_key]
                        if sub_sub_value != [np.nan] and existing_value != [np.nan]:
                            # Both values are not [np.nan], so merge and sort them
                            merged_dict[main_key][sub_key][sub_sub_key] = sorted(existing_value + sub_sub_value)
                        elif sub_sub_value != [np.nan]:
                            # The new value is not [np.nan], but the old one is, so replace it with the new value
                            merged_dict[main_key][sub_key][sub_sub_key] = sorted(sub_sub_value)
                        # If the new value is [np.nan], keep the existing value unchanged
                    else:
                        # The key does not exist; add it directly (retain all sessions even if the value is [np.nan])
                        if sub_sub_value != [np.nan]:
                            merged_dict[main_key][sub_key][sub_sub_key] = sorted(sub_sub_value)
                        else:
                            merged_dict[main_key][sub_key][sub_sub_key] = [np.nan]
    return merged_dict

def merge_bhvr_intervals(data, gap_threshold=0):
    """
    Merge overlapping intervals or intervals separated by less than the threshold.

    Parameters:
    -----------
    data : dict or list
        Nested dictionary or list of interval tuples [(start, end), ...]
    gap_threshold : int
        Gap threshold in frames; default is 0 (only overlapping intervals are merged).
        When the gap between two intervals is <= gap_threshold, they are merged.
        For example, with gap_threshold=30, [(0, 10), (15, 20)] (gap 4 frames) would be merged into [(0, 20)].
        With gap_threshold=3, the same example would not be merged.

    Returns:
    --------
    dict or list
        Merged data while preserving the original nested structure.

    Examples:
    ---------
    >>> # Merge overlapping intervals
    >>> merge_overlap([(0, 10), (5, 15), (20, 30)])
    [(0, 15), (20, 30)]

    >>> # Merge intervals with a gap <= 3 frames
    >>> merge_overlap([(0, 10), (12, 20), (25, 30)], gap_threshold=3)
    [(0, 20), (25, 30)]

    >>> # Process nested dictionaries
    >>> data = {'s': {'D': {'B1MD': [(0, 10), (12, 20)]}}}
    >>> merge_overlap(data, gap_threshold=3)
    {'s': {'D': {'B1MD': [(0, 20)]}}}
    """
    if isinstance(data, dict):
        filtered_dict = {k: v for k, v in ((k, merge_bhvr_intervals(v, gap_threshold)) for k, v in data.items()) if v is not None}
        return filtered_dict if filtered_dict else np.nan
    elif isinstance(data, list):
        # Handle cases containing np.nan
        if len(data) > 0 and not isinstance(data[0], tuple):
            # If the list contains np.nan or other non-tuple data, return it directly
            return data
        
        sorted_data = sorted(data, key=lambda x: x[0])
        merged_data = []
        for tuple_data in sorted_data:
            if not merged_data:
                # First interval: add it directly
                merged_data.append(tuple_data)
            elif tuple_data[0] - merged_data[-1][1] > gap_threshold:
                # The current interval starts after the previous one by more than the threshold, so do not merge
                merged_data.append(tuple_data)
            else:
                # Overlapping or gap <= threshold: merge
                merged_data[-1] = (merged_data[-1][0], max(merged_data[-1][1], tuple_data[1]))
        return merged_data


def filter_by_start_time(data, t_range):
    if isinstance(data, dict):
        filtered_dict = {}
        for k, v in data.items():
            filtered_v = filter_by_start_time(v, t_range)
            if filtered_v is not None and filtered_v != [np.nan]:
                filtered_dict[k] = filtered_v
        return filtered_dict if filtered_dict else np.nan
    elif isinstance(data, list):
        # Filter bouts whose start time falls within the range
        filtered_list = [
            bout for bout in data 
            if isinstance(bout, tuple) and t_range[0] <= bout[0] <= t_range[1]
        ]
        return filtered_list if filtered_list else [np.nan]
    elif isinstance(data, tuple):
        start, end = data
        # Only check whether the start time falls within the range and keep the full bout
        if t_range[0] <= start <= t_range[1]:
            return (start, end)
        return None
    return data

def analyze_independence(sub_dom_raw, labels, title="Independence Analysis", data_type="count"):

    print("="*70)
    print(f"{title}")
    print("="*70)
    
    n = sub_dom_raw.shape[0]
    
    # ============================================================
    # Step 1: Calculate the expected distribution (assuming rows and columns are independent)
    # ============================================================
    print("\n[Step 1: Calculate the expected independent distribution]")
    
    # Row and column marginals
    sub = np.sum(sub_dom_raw, axis=1)  # Marginal distribution for Submissive
    dom = np.sum(sub_dom_raw, axis=0)  # Marginal distribution for Dominant

    # Marginal probabilities
    sub_perc = sub / np.sum(sub)
    dom_perc = dom / np.sum(dom)
    
    # Expected percentages under the independence assumption
    sub_dom_perc = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sub_dom_perc[i, j] = sub_perc[i] * dom_perc[j]
    
    # Observed percentages
    sub_dom_raw_perc = sub_dom_raw / np.sum(sub_dom_raw)
    
    # ============================================================
    # Visualization 1: Observed distribution vs expected distribution
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Observed distribution
    if data_type == "count":
        sns.heatmap(sub_dom_raw, annot=True, fmt='.0f', cmap='viridis', 
                    ax=axes[0], cbar_kws={'label': 'count'},
                    xticklabels=labels, yticklabels=labels)
        axes[0].set_title('Observed Distribution (counts)')
    else:
        sns.heatmap(sub_dom_raw_perc * 100, annot=True, fmt='.2f', cmap='viridis', 
                    ax=axes[0], cbar_kws={'label': '%'},
                    xticklabels=labels, yticklabels=labels)
        axes[0].set_title('Observed Distribution (%)')
    axes[0].set_xlabel('Dom')
    axes[0].set_ylabel('Sub')
    axes[0].invert_yaxis()
    
    # Expected distribution
    sns.heatmap(sub_dom_perc * 100, annot=True, fmt='.2f', cmap='viridis', 
                ax=axes[1], cbar_kws={'label': '%'},
                xticklabels=labels, yticklabels=labels)
    axes[1].set_title('Expected Distribution (Independence, %)')
    axes[1].set_xlabel('Dom')
    axes[1].set_ylabel('Sub')
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.show()
    
    # ============================================================
    # Visualization 2: Difference analysis
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Absolute difference
    diff_perc = (sub_dom_raw_perc - sub_dom_perc)
    vmax = np.abs(diff_perc).max()
    
    diff_abs_df = pd.DataFrame(diff_perc * 100, index=labels, columns=labels)
    sns.heatmap(diff_abs_df, annot=True, fmt='.2f', cmap='bwr', 
                vmax=vmax*100, vmin=-vmax*100, center=0, ax=axes[0],
                cbar_kws={'label': '% difference'})
    axes[0].set_title('Difference: Observed - Expected (%)')
    axes[0].set_xlabel('Dom')
    axes[0].set_ylabel('Sub')
    axes[0].invert_yaxis()
    
    # Standardized difference
    diff_rel = (sub_dom_raw_perc - sub_dom_perc) / (sub_dom_perc + sub_dom_raw_perc + 1e-10)
    diff_rel_df = pd.DataFrame(diff_rel, index=labels, columns=labels)
    sns.heatmap(diff_rel_df, annot=True, fmt='.2f', cmap='bwr', 
                vmax=1, vmin=-1, center=0, ax=axes[1],
                cbar_kws={'label': 'normalized difference'})
    axes[1].set_title('Normalized Difference')
    axes[1].set_xlabel('Dom')
    axes[1].set_ylabel('Sub')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.show()
    
    # ============================================================
    # TEST 1: Chi-Square Test of Independence (Overall)
    # ============================================================
    chi2, p_value, dof, expected = chi2_contingency(sub_dom_raw)

    print("="*50)
    print("CHI-SQUARE TEST OF INDEPENDENCE")
    print("="*50)
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"P-value: {p_value:.6e}")
    print(f"Degrees of freedom: {dof}")

    if p_value < 0.05:
        print("\n✓ Significant difference (p < 0.05)")
        print("  Observed and expected distributions are different")
    else:
        print("\n✗ Not significant (p >= 0.05)")
        print("  No evidence of difference")

    # ============================================================
    # TEST 2: Standardized Residuals (Which cells are different?)
    # ============================================================
    print("\n" + "="*50)
    print("STANDARDIZED RESIDUALS (Cell-wise significance)")
    print("="*50)

    std_residuals = (sub_dom_raw - expected) / np.sqrt(expected)
    print("Standardized residuals:")
    print(std_residuals)
    print("\n|residual| > 2 indicates significant deviation at α=0.05")

    significant_cells = np.abs(std_residuals) > 2
    print("\nSignificant cells:")
    for i in range(3):
        for j in range(3):
            if significant_cells[i,j]:
                print(f"  Cell ({i},{j}): residual = {std_residuals[i,j]:.3f}")

    # ============================================================
    # TEST 3: Cramér's V (Effect size)
    # ============================================================
    print("\n" + "="*50)
    print("EFFECT SIZE (CRAMÉR'S V)")
    print("="*50)

    def cramers_v(contingency_table):
        n = np.sum(contingency_table)
        chi2 = chi2_contingency(contingency_table)[0]
        min_dim = min(contingency_table.shape) - 1
        return np.sqrt(chi2 / (n * min_dim))

    v = cramers_v(sub_dom_raw)
    print(f"Cramér's V: {v:.4f}")

    if v < 0.1:
        print("Effect: Very small")
    elif v < 0.3:
        print("Effect: Small")
    elif v < 0.5:
        print("Effect: Medium")
    else:
        print("Effect: Large")
        
    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Chi-square p-value: {p_value:.6f}")
    print(f"Cramér's V: {v:.4f}")
    print(f"Significant cells: {np.sum(significant_cells)} out of 9")

print("✓ Independence test framework loaded successfully")


def check_if_bhvr_reduce_freezing_duration(
    behavior_labels_dict,
    target_behavior,
    event_type='offset',
    t_range=(0, 8900),
    data_type='ret',
    fps=30,
    window_before=10,
    window_after=10,
    window_before_stats=1,  # Window for freezing-time statistics (seconds)
    window_after_stats=1,   # Window for freezing-time statistics (seconds)
    n_permutations=1000,
    n_random_total=1000,
    all_bhvrs=None,
    bhvr_abbrev_dict=None,
    bhvr_color_dict=None,
    excluded_behaviors=None,
    composite_behaviors=None,
    save_prefix='',
    trial_id_format='rat',  # 'rat' or 'looming'
    figsize=(10, 6),
    grid=True,
    legend=True,
    save_fig_dir=None,
    title=None,
):
    """
    Check whether a target behavior reduces the partner's freezing duration.

    Parameters:
    --------
    behavior_labels_dict : dict
        Behavior label dictionary
    target_behavior : str
        Target behavior name (e.g. 'approach_partner')
    event_type : str
        Event type, 'onset' or 'offset'
    t_range : tuple
        Time range (min, max)
    data_type : str
        Data type, 'ret' or 'lst'
    fps : int
        Frame rate
    window_before : int
        Window before offset/onset for plotting (seconds)
    window_after : int
        Window after offset/onset for plotting (seconds)
    window_before_stats : float
        Window before offset/onset used for freezing-duration statistics (seconds)
    window_after_stats : float
        Window after offset/onset used for freezing-duration statistics (seconds)
    n_permutations : int
        Number of permutations for the permutation test
    n_random_total : int
        Total number of random time points to generate
    all_bhvrs : list
        List of all behaviors
    bhvr_abbrev_dict : dict
        Behavior abbreviation dictionary
    bhvr_color_dict : dict
        Behavior color dictionary
    excluded_behaviors : list
        Behaviors to exclude from plotting
    composite_behaviors : list of tuple
        Composite behaviors
    filter_method : str
        Method for filter_in_range
    save_prefix : str
        Prefix for saved files
    trial_id_format : str
        Trial ID format, 'rat' (e.g. 'B10MD') or 'looming' (e.g. 'B10MDT3')
    random_seed : int
        Random seed

    Returns:
    --------
    results : dict
        Dictionary containing the statistical results
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from analyze_data_utils import filter_dict_data, filter_in_range
    
    np.random.seed(42)
    
    # ========== Internal helper: get the partner trial ID ==========
    def get_partner_trial_id(trial_id, rank, trial_id_format):
        """Get the partner trial ID based on the trial_id format."""
        partner_rank = 'S' if rank == 'D' else 'D'
        
        if trial_id_format == 'rat':
            return trial_id.split('M')[0] + 'M' + partner_rank
        elif trial_id_format == 'looming':
            if 'MD' in trial_id:
                return trial_id.replace('MD', 'MS')
            elif 'MS' in trial_id:
                return trial_id.replace('MS', 'MD')
            else:
                return None
        else:
            raise ValueError(f"Unknown trial_id_format: {trial_id_format}")
    
    # ========== Internal helper: filter bouts where the partner was freezing in the previous frame ==========
    def filter_bouts_with_partner_freezing(bhvr_data, freezing_data, trial_id_format):
        """Filter behavior bouts for which the partner was freezing in the previous frame."""
        filtered_data = {}
        
        for group in bhvr_data:
            filtered_data[group] = {}
            for rank in bhvr_data[group]:
                filtered_data[group][rank] = {}
                
                # Check whether the value is a dictionary
                if not isinstance(bhvr_data[group][rank], dict):
                    continue
                
                for trial_id, bouts in bhvr_data[group][rank].items():
                    # Get the partner trial ID
                    partner_trial_id = get_partner_trial_id(trial_id, rank, trial_id_format)
                    if partner_trial_id is None:
                        continue
                    
                    # Get the partner freezing data
                    partner_rank = 'S' if rank == 'D' else 'D'
                    partner_freezing = freezing_data.get(group, {}).get(partner_rank, {}).get(partner_trial_id, [])
                    
                    # Check data validity
                    if not isinstance(partner_freezing, list):
                        continue
                    if not partner_freezing or (len(partner_freezing) == 1 and 
                        isinstance(partner_freezing[0], (int, float)) and np.isnan(partner_freezing[0])):
                        continue
                    
                    # Filter matching bouts
                    filtered_bouts = []
                    for bout in bouts:
                        if not isinstance(bout, (list, tuple, np.ndarray)) or len(bout) < 2:
                            continue
                        
                        bout_start, bout_end = float(bout[0]), float(bout[1])
                        
                        # Determine the frame to check based on event_type
                        if event_type == 'offset':
                            check_frame = bout_end - 1
                        elif event_type == 'onset':
                            check_frame = bout_start - 1
                        else:
                            check_frame = bout_end - 1  # Default to offset
                        
                        # Check whether the frame falls within the partner's freezing bout
                        is_freezing = False
                        for f_bout in partner_freezing:
                            if not isinstance(f_bout, (list, tuple, np.ndarray)) or len(f_bout) < 2:
                                continue
                            f_start, f_end = float(f_bout[0]), float(f_bout[1])
                            
                            if f_start <= check_frame <= f_end:
                                is_freezing = True
                                break
                        
                        if is_freezing:
                            filtered_bouts.append(bout)
                    
                    if filtered_bouts:
                        filtered_data[group][rank][trial_id] = filtered_bouts
        
        return filtered_data
    
    # ========== Internal helper: calculate freezing time before and after a time point ==========
    def calculate_freezing_around_timepoint(timepoint, freezing_bouts, window_before_frames, window_after_frames, fps):
        """Calculate freezing time within the window before and after a specified time point."""
        # Before window
        before_window_start = timepoint - window_before_frames
        before_window_end = timepoint
        freezing_before = 0
        
        for f_bout in freezing_bouts:
            if not isinstance(f_bout, (list, tuple, np.ndarray)) or len(f_bout) < 2:
                continue
            f_start, f_end = float(f_bout[0]), float(f_bout[1])
            overlap_start = max(f_start, before_window_start)
            overlap_end = min(f_end, before_window_end)
            if overlap_start < overlap_end:
                freezing_before += (overlap_end - overlap_start)
        
        # After window
        after_window_start = timepoint
        after_window_end = timepoint + window_after_frames
        freezing_after = 0
        
        for f_bout in freezing_bouts:
            if not isinstance(f_bout, (list, tuple, np.ndarray)) or len(f_bout) < 2:
                continue
            f_start, f_end = float(f_bout[0]), float(f_bout[1])
            overlap_start = max(f_start, after_window_start)
            overlap_end = min(f_end, after_window_end)
            if overlap_start < overlap_end:
                freezing_after += (overlap_end - overlap_start)
        
        return freezing_before / fps, freezing_after / fps
    
    # ========== 1. Load data and filter it ==========
    print(f"\n{'='*60}")
    print(f"analyze how {target_behavior} {event_type} affects partner freezing")
    print(f"{'='*60}\n")
    
    bhvr_data = filter_dict_data(behavior_labels_dict, target_behavior)
    # bhvr_data = filter_in_range(bhvr_data, t_range=t_range, method=filter_method)
    bhvr_data = filter_by_start_time(bhvr_data, t_range=t_range)

    freezing_data = filter_dict_data(behavior_labels_dict, 'freezing')
    # freezing_data = filter_in_range(freezing_data, t_range=t_range, method=filter_method)
    freezing_data = filter_by_start_time(freezing_data, t_range=t_range)
    
    # Filter bouts where the partner was freezing in the previous frame
    print("1. Filter out the bouts where the -1 frame's partner behavior is freezing.")
    filtered_bhvr_data = filter_bouts_with_partner_freezing(bhvr_data, freezing_data, trial_id_format)
    
    # Summarize the filtered results
    total_filtered_bouts = sum(
        len(bouts) 
        for group in filtered_bhvr_data 
        for rank in filtered_bhvr_data[group] 
        for bouts in filtered_bhvr_data[group][rank].values()
    )
    print(f"There are {total_filtered_bouts} bouts\n")
    
    # ========== 2. Plot the filtered behavior distribution ==========
    print("2. Draw filtered behavior distribution")
    # 1. Unit conversion: 5.08 cm = 2.0 inches
    target_inch = 5.08 / 2.54  

    # 2. Reserve margins for axis labels and ticks (in inches); adjust based on font size if needed
    left_margin = 0.4    # Left margin for the y-axis label
    right_margin = 0.1   # Right margin
    bottom_margin = 0.3  # Bottom margin for the x-axis label
    top_margin = 0.1     # Top margin

    # 3. Compute the final canvas size
    fig_w = target_inch + left_margin + right_margin
    fig_h = target_inch + bottom_margin + top_margin

    plot_behavior_distribution_around_event(
        reference_behavior_data=filtered_bhvr_data,
        all_behavior_labels_dict=behavior_labels_dict,
        data_type=data_type,
        reference_behavior=target_behavior,
        event_type=event_type,
        if_partner=True,
        window_before=window_before,
        window_after=window_after,
        fps=fps,
        bin_size_frames=1,
        **{f'{data_type}_all_bhvrs': all_bhvrs} if all_bhvrs else {},
        **{f'{data_type}_bhvr_abbrev_dict': bhvr_abbrev_dict} if bhvr_abbrev_dict else {},
        **{f'{data_type}_bhvr_color_dict': bhvr_color_dict} if bhvr_color_dict else {},
        excluded_behaviors=excluded_behaviors or [],
        composite_behaviors=composite_behaviors,
        figsize=(fig_w, fig_h),
        ylabel='Freezing occupancy (%)',
        xlabel='Time (s)',
        grid=grid,
        legend=legend,
        save_fig_dir=save_fig_dir,
        title=title,
    )
    
    # ========== 3. Collect freezing-time changes for the real data ==========
    print("3. calculate freezing time before and after the event for real data...")
    window_before_frames = window_before_stats * fps
    window_after_frames = window_after_stats * fps
    
    real_data = []
    for group in filtered_bhvr_data:
        for rank in filtered_bhvr_data[group]:
            partner_rank = 'S' if rank == 'D' else 'D'
            
            for trial_id, bouts in filtered_bhvr_data[group][rank].items():
                partner_trial_id = get_partner_trial_id(trial_id, rank, trial_id_format)
                if partner_trial_id is None:
                    continue
                
                partner_freezing = freezing_data.get(group, {}).get(partner_rank, {}).get(partner_trial_id, [])
                if not isinstance(partner_freezing, list) or not partner_freezing:
                    continue
                
                for bout in bouts:
                    if not isinstance(bout, (list, tuple, np.ndarray)) or len(bout) < 2:
                        continue
                    
                    bout_start, bout_end = float(bout[0]), float(bout[1])
                    timepoint = bout_end if event_type == 'offset' else bout_start
                    
                    freezing_before_sec, freezing_after_sec = calculate_freezing_around_timepoint(
                        timepoint, partner_freezing, window_before_frames, window_after_frames, fps
                    )
                    
                    real_data.append({
                        'group': 'real',
                        'trial_id': trial_id,
                        'timepoint': timepoint,
                        'freezing_before': freezing_before_sec,
                        'freezing_after': freezing_after_sec,
                        'difference': freezing_before_sec - freezing_after_sec
                    })
    
    print(f"There are {len(real_data)} real data\n")
    
    # ========== 4. Generate random time-point data ==========
    print(f"4. generate {n_random_total} random timepoints (previous frame's partner is also freezing)...")
    
    # Collect all possible trial information
    trial_info_list = []
    for group in bhvr_data:
        for rank in bhvr_data[group]:
            if not isinstance(bhvr_data[group][rank], dict):
                continue
            
            partner_rank = 'S' if rank == 'D' else 'D'
            
            for trial_id in bhvr_data[group][rank].keys():
                partner_trial_id = get_partner_trial_id(trial_id, rank, trial_id_format)
                if partner_trial_id is None:
                    continue
                
                partner_freezing = freezing_data.get(group, {}).get(partner_rank, {}).get(partner_trial_id, [])
                
                if not isinstance(partner_freezing, list) or not partner_freezing:
                    continue
                if len(partner_freezing) == 1 and isinstance(partner_freezing[0], (int, float)) and np.isnan(partner_freezing[0]):
                    continue
                
                trial_info_list.append({
                    'group': group,
                    'rank': rank,
                    'trial_id': trial_id,
                    'partner_trial_id': partner_trial_id,
                    'partner_freezing': partner_freezing
                })
    
    # Generate random time points
    random_timepoints_data = {}
    random_data = []
    generated_count = 0
    max_total_attempts = n_random_total * 100
    attempt_count = 0
    
    while generated_count < n_random_total and attempt_count < max_total_attempts:
        attempt_count += 1
        
        # Randomly select a trial
        trial_info = trial_info_list[np.random.randint(0, len(trial_info_list))]
        
        # Randomly generate a time point within the time range
        random_timepoint = int(np.random.randint(t_range[0], t_range[1]))
        check_frame = random_timepoint - 1
        
        # Check whether the previous frame falls within the partner's freezing bout
        is_freezing = False
        for f_bout in trial_info['partner_freezing']:
            if not isinstance(f_bout, (list, tuple, np.ndarray)) or len(f_bout) < 2:
                continue
            f_start, f_end = float(f_bout[0]), float(f_bout[1])
            
            if f_start <= check_frame <= f_end:
                is_freezing = True
                break
        
        if is_freezing:
            # Calculate freezing time before and after the random time point
            freezing_before_sec, freezing_after_sec = calculate_freezing_around_timepoint(
                random_timepoint, trial_info['partner_freezing'], 
                window_before_frames, window_after_frames, fps
            )
            
            random_data.append({
                'group': 'random',
                'trial_id': trial_info['trial_id'],
                'timepoint': random_timepoint,
                'freezing_before': freezing_before_sec,
                'freezing_after': freezing_after_sec,
                'difference': freezing_before_sec - freezing_after_sec
            })
            
            # Save to random_timepoints_data for plotting
            group = trial_info['group']
            rank = trial_info['rank']
            trial_id = trial_info['trial_id']
            
            if group not in random_timepoints_data:
                random_timepoints_data[group] = {}
            if rank not in random_timepoints_data[group]:
                random_timepoints_data[group][rank] = {}
            if trial_id not in random_timepoints_data[group][rank]:
                random_timepoints_data[group][rank][trial_id] = []
            
            random_start = random_timepoint - 30
            random_end = random_timepoint
            random_timepoints_data[group][rank][trial_id].append((int(random_start), int(random_end)))
            
            generated_count += 1
    
    print(f"Generated {generated_count} random timepoints after {attempt_count} attempts\n")
    
    ## ========== 5. Plot the behavior distribution at random time points ==========
    # print("5. draw random timepoints partner behavior distribution...")
    # plot_behavior_distribution_around_event(
    #     reference_behavior_data=random_timepoints_data,
    #     all_behavior_labels_dict=behavior_labels_dict,
    #     data_type=data_type,
    #     reference_behavior='random_timepoint',
    #     event_type=event_type,
    #     if_partner=True,
    #     window_before=window_before,
    #     window_after=window_after,
    #     fps=fps,
    #     bin_size_frames=1,
    #     **{f'{data_type}_all_bhvrs': all_bhvrs} if all_bhvrs else {},
    #     **{f'{data_type}_bhvr_abbrev_dict': bhvr_abbrev_dict} if bhvr_abbrev_dict else {},
    #     **{f'{data_type}_bhvr_color_dict': bhvr_color_dict} if bhvr_color_dict else {},
    #     excluded_behaviors=excluded_behaviors or [],
    #     composite_behaviors=composite_behaviors
    # )    
    
    # ========== 6. Permutation Test ==========
    print(f"6. perform Permutation Test ({n_permutations} permutations)...")
    
    df_real = pd.DataFrame(real_data)
    df_random = pd.DataFrame(random_data)
    
    real_diffs = df_real['difference'].values
    random_diffs = df_random['difference'].values
    
    observed_mean_diff = np.mean(real_diffs) - np.mean(random_diffs)
    observed_median_diff = np.median(real_diffs) - np.median(random_diffs)
    
    # Run the permutation test
    n_real = len(real_diffs)
    n_random = len(random_diffs)
    pooled_data = np.concatenate([real_diffs, random_diffs])
    
    permuted_mean_diffs = []
    permuted_median_diffs = []
    
    for i in range(n_permutations):
        shuffled = np.random.permutation(pooled_data)
        perm_real = shuffled[:n_real]
        perm_random = shuffled[n_real:]
        
        perm_mean_diff = np.mean(perm_real) - np.mean(perm_random)
        perm_median_diff = np.median(perm_real) - np.median(perm_random)
        
        permuted_mean_diffs.append(perm_mean_diff)
        permuted_median_diffs.append(perm_median_diff)
    
    permuted_mean_diffs = np.array(permuted_mean_diffs)
    permuted_median_diffs = np.array(permuted_median_diffs)
    
    # Calculate p-values
    p_value_mean = (np.sum(permuted_mean_diffs >= observed_mean_diff) + 1) / (n_permutations + 1)
    p_value_median = (np.sum(permuted_median_diffs >= observed_median_diff) + 1) / (n_permutations + 1)
    
    
    # ========== 7. Print the statistical results ==========
    print(f"{'='*30}")
    print(f"real ±1 s mean freezing reduction: {np.mean(real_diffs):.4f} sec")
    print(f"random  ±1 s mean freezing reduction: {np.mean(random_diffs):.4f} sec")
    print(f"observed mean difference: {observed_mean_diff:.4f} sec")
    print(f"p-value for mean difference: {p_value_mean:.4f}")
    print(f"\nreal group median reduction: {np.median(real_diffs):.4f} sec")
    print(f"random group median reduction: {np.median(random_diffs):.4f} sec")
    print(f"observed median difference: {observed_median_diff:.4f} sec")
    print(f"p-value for median difference: {p_value_median:.4f}")
    print(f"\nConclusion: ", end='')

    
    # ========== 8. Visualization ==========
    print("8. visualize the results...")
    fig, axes = plt.subplots(2, 1, figsize=(5, 8))
    
    # Violin plot
    ax1 = axes[0]
    parts = ax1.violinplot([real_diffs, random_diffs], positions=[1, 2], 
                           showmeans=False, showmedians=False, widths=0.4)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(['orange', 'blue'][i])
        pc.set_alpha(0.7)
    
    for partname in ('cbars', 'cmins', 'cmaxes'):
        if partname in parts:
            parts[partname].set_visible(False)
    
    for i, (data, pos, color) in enumerate([(real_diffs, 1, 'orange'), (random_diffs, 2, 'blue')]):
        q25 = np.percentile(data, 25)
        q75 = np.percentile(data, 75)
        median = np.median(data)
        mean = np.mean(data)
        
        ax1.plot([pos, pos], [q25, q75], color='black', linewidth=4, zorder=3)
        ax1.plot([pos-0.08, pos+0.08], [median, median], color='white', linewidth=2.5, zorder=4)
        ax1.scatter([pos], [mean], color='white', s=50, zorder=5, edgecolors='black', linewidths=1)
    
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels([f'Real {target_behavior.upper()}', 'Random'])
    ax1.set_xlim(0.5, 2.5)
    ax1.set_ylabel('Freezing Reduction (sec)', fontsize=12)
    ax1.set_title('Distribution Comparison', fontsize=13)
    ax1.grid(alpha=0.3, axis='y')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Permutation distribution
    ax2 = axes[1]
    ax2.hist(permuted_mean_diffs, bins=50, alpha=0.7, color='gray', edgecolor='black')
    ax2.axvline(observed_mean_diff, color='red', linestyle='--', linewidth=2.5, 
               label=f'Observed: {observed_mean_diff:.4f}s')
    ax2.set_xlabel('Mean Difference (Real - Random, sec)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'Permutation Distribution\np = {p_value_mean:.4f}', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # ========== 9. Save the data ==========
    if save_prefix:
        print("9. saveing results...")
        df_combined = pd.concat([df_real, df_random], ignore_index=True)
        data_path = f'{save_prefix}_{target_behavior}_{event_type}_freezing_data.csv'
        df_combined.to_csv(data_path, index=False)
        
        stats_summary = pd.DataFrame({
            'metric': ['Mean', 'Median', 'Std', 'N'],
            'real': [np.mean(real_diffs), np.median(real_diffs), np.std(real_diffs), len(real_diffs)],
            'random': [np.mean(random_diffs), np.median(random_diffs), np.std(random_diffs), len(random_diffs)],
            'difference': [observed_mean_diff, observed_median_diff, np.nan, np.nan],
            'p_value': [p_value_mean, p_value_median, np.nan, np.nan]
        })
        stats_path = f'{save_prefix}_{target_behavior}_{event_type}_stats.csv'
        stats_summary.to_csv(stats_path, index=False)
        
        print(f"data saved to: {data_path}")
        print(f"statistics results saved to: {stats_path}\n")
    

def plot_violin(
    behavior_labels_dict,
    data_sources=[
    ('approach_partner', 'offset', 'AP offset', 'orange'),
    ('sniff_partner', 'onset', 'SP onset', 'green'),
    ('freezing', 'offset', 'F offset', 'blue'),
    ],
    t_range=None,
    fps=30,
    window_before_stats=1,
    window_after_stats=1,
    trial_id_format=None, # 'rat' or 'lst'
    ranks=None, # None=all, 'D', 'S'
    jitter=False,
    n_random_total=1000, # Total number of random time points to generate
    n_permutations=1000, # Number of permutation test iterations; 0 skips the test
    target_behavior='freezing', # Partner behavior used to compute reduction
    ylabel=None,
    title='Violin Plot',
    save_fig_dir=None,
    ):

    if ylabel is None:
        ylabel = f'Δ {target_behavior.capitalize()} time (s)'

    window_before_frames = window_before_stats * fps
    window_after_frames  = window_after_stats  * fps
    allowed_ranks = set([ranks] if isinstance(ranks, str) else ranks) if ranks else None

    # ---------- helpers ----------

    def get_partner_trial_id(trial_id, rank):
        if trial_id_format in ('rat', 'lst'):
            if rank == 'D':
                return trial_id.replace('MD', 'MS')
            else:
                return trial_id.replace('MS', 'MD')
        elif trial_id_format == 'looming':
            if 'MD' in trial_id:
                return trial_id.replace('MD', 'MS')
            elif 'MS' in trial_id:
                return trial_id.replace('MS', 'MD')
            return None
        else:
            raise ValueError(f"Unknown trial_id_format: {trial_id_format}")

    def filter_bouts_with_partner_behavior(bhvr_data, target_bhvr_data, event_type):
        filtered = {}
        for group in ['p']:
            filtered[group] = {}
            for rank in bhvr_data[group]:
                if allowed_ranks and rank not in allowed_ranks:
                    filtered[group][rank] = {}
                    continue
                filtered[group][rank] = {}
                if not isinstance(bhvr_data[group][rank], dict):
                    continue
                for trial_id, bouts in bhvr_data[group][rank].items():
                    partner_trial_id = get_partner_trial_id(trial_id, rank)
                    if partner_trial_id is None:
                        continue
                    partner_rank = 'S' if rank == 'D' else 'D'
                    pb = target_bhvr_data.get(group, {}).get(partner_rank, {}).get(partner_trial_id, [])
                    if not isinstance(pb, list) or not pb:
                        continue
                    if len(pb) == 1 and isinstance(pb[0], (int, float)) and np.isnan(pb[0]):
                        continue
                    ok_bouts = []
                    for bout in bouts:
                        if not isinstance(bout, (list, tuple, np.ndarray)) or len(bout) < 2:
                            continue
                        check = (float(bout[1]) if event_type == 'offset' else float(bout[0])) - 1
                        if any(
                            isinstance(f, (list, tuple, np.ndarray)) and len(f) >= 2
                            and float(f[0]) <= check <= float(f[1])
                            for f in pb
                        ):
                            ok_bouts.append(bout)
                    if ok_bouts:
                        filtered[group][rank][trial_id] = ok_bouts
        return filtered

    def calculate_behavior_around_timepoint(timepoint, target_bouts, wb_frames, wa_frames):
        bhvr_before = sum(
            min(float(f[1]), timepoint) - max(float(f[0]), timepoint - wb_frames)
            for f in target_bouts
            if isinstance(f, (list, tuple, np.ndarray)) and len(f) >= 2
            and max(float(f[0]), timepoint - wb_frames) < min(float(f[1]), timepoint)
        ) / fps
        bhvr_after = sum(
            min(float(f[1]), timepoint + wa_frames) - max(float(f[0]), timepoint)
            for f in target_bouts
            if isinstance(f, (list, tuple, np.ndarray)) and len(f) >= 2
            and max(float(f[0]), timepoint) < min(float(f[1]), timepoint + wa_frames)
        ) / fps
        return bhvr_before, bhvr_after

    def compute_diffs(filtered_bhvr_data, target_bhvr_data, event_type):
        diffs = []
        for group in filtered_bhvr_data:
            for rank in filtered_bhvr_data[group]:
                if allowed_ranks and rank not in allowed_ranks:
                    continue
                partner_rank = 'S' if rank == 'D' else 'D'
                for trial_id, bouts in filtered_bhvr_data[group][rank].items():
                    partner_trial_id = get_partner_trial_id(trial_id, rank)
                    if partner_trial_id is None:
                        continue
                    pb = target_bhvr_data.get(group, {}).get(partner_rank, {}).get(partner_trial_id, [])
                    if not isinstance(pb, list) or not pb:
                        continue
                    for bout in bouts:
                        if not isinstance(bout, (list, tuple, np.ndarray)) or len(bout) < 2:
                            continue
                        tp = float(bout[1]) if event_type == 'offset' else float(bout[0])
                        before, after = calculate_behavior_around_timepoint(
                            tp, pb, window_before_frames, window_after_frames)
                        diffs.append(before - after)
        return np.array(diffs)

    def generate_random_diffs(bhvr_data, target_bhvr_data, n_random):
        """Randomly sample time points within t_range (requiring that the partner was in the target behavior on the previous frame), matching the reference code logic."""
        trial_info_list = []
        for group in bhvr_data:
            for rank in bhvr_data[group]:
                if not isinstance(bhvr_data[group][rank], dict):
                    continue
                if allowed_ranks and rank not in allowed_ranks:
                    continue
                partner_rank = 'S' if rank == 'D' else 'D'
                for trial_id in bhvr_data[group][rank].keys():
                    partner_trial_id = get_partner_trial_id(trial_id, rank)
                    if partner_trial_id is None:
                        continue
                    pb = target_bhvr_data.get(group, {}).get(partner_rank, {}).get(partner_trial_id, [])
                    if not isinstance(pb, list) or not pb:
                        continue
                    if len(pb) == 1 and isinstance(pb[0], (int, float)) and np.isnan(pb[0]):
                        continue
                    trial_info_list.append({'trial_id': trial_id, 'partner_target_bhvr': pb})

        if not trial_info_list:
            return np.array([])

        random_diffs = []
        generated_count = 0
        max_attempts = n_random * 100
        attempt_count = 0

        while generated_count < n_random and attempt_count < max_attempts:
            attempt_count += 1
            info = trial_info_list[np.random.randint(0, len(trial_info_list))]
            tp = int(np.random.randint(t_range[0], t_range[1]))
            check_frame = tp - 1
            is_in_behavior = any(
                isinstance(f, (list, tuple, np.ndarray)) and len(f) >= 2
                and float(f[0]) <= check_frame <= float(f[1])
                for f in info['partner_target_bhvr']
            )
            if is_in_behavior:
                before, after = calculate_behavior_around_timepoint(
                    tp, info['partner_target_bhvr'], window_before_frames, window_after_frames)
                random_diffs.append(before - after)
                generated_count += 1

        return np.array(random_diffs)

    def run_permutation_test(real_diffs, random_diffs, n_perm):
        """Two-sample permutation test, matching the reference code exactly."""
        observed_mean_diff = np.mean(real_diffs) - np.mean(random_diffs)
        n_real = len(real_diffs)
        pooled = np.concatenate([real_diffs, random_diffs])
        permuted_mean_diffs = []
        for _ in range(n_perm):
            shuffled = np.random.permutation(pooled)
            permuted_mean_diffs.append(np.mean(shuffled[:n_real]) - np.mean(shuffled[n_real:]))
        permuted_mean_diffs = np.array(permuted_mean_diffs)
        p_value = (np.sum(permuted_mean_diffs >= observed_mean_diff) + 1) / (n_perm + 1)
        return p_value, observed_mean_diff, permuted_mean_diffs

    def pval_to_stars(p):
        if p < 0.001:   return '***'
        elif p < 0.01:  return '**'
        elif p < 0.05:  return '*'
        else:           return ''

    # ---------- main ----------

    target_bhvr_data = filter_dict_data(behavior_labels_dict, target_behavior)
    target_bhvr_data = filter_by_start_time(target_bhvr_data, t_range=t_range)

    violin_data = []
    for behavior, event_type, label, color in data_sources:
        bhvr_data = filter_dict_data(behavior_labels_dict, behavior)
        bhvr_filtered = filter_by_start_time(bhvr_data, t_range=t_range)
        filtered  = filter_bouts_with_partner_behavior(bhvr_filtered, target_bhvr_data, event_type)
        real_diffs = compute_diffs(filtered, target_bhvr_data, event_type)
        print(f"{label}: {len(real_diffs)} bouts")
        if len(real_diffs) == 0:
            print(f"  [WARNING] '{label}' has no data, skipped.")
            continue

        stars = ''
        if n_permutations > 0:
            rand_diffs = generate_random_diffs(bhvr_data, target_bhvr_data, n_random_total)
            print(f"  generated {len(rand_diffs)} random timepoints")
            if len(rand_diffs) > 0:
                p_val, obs_diff, _ = run_permutation_test(real_diffs, rand_diffs, n_permutations)
                stars = pval_to_stars(p_val)
                print(f"  observed mean diff = {obs_diff:.4f} sec,  p = {p_val:.4f}  {stars}")
            else:
                print(f"  [WARNING] no random timepoints generated, skipping test.")

        violin_data.append((real_diffs, label, color, stars))

    if not violin_data:
        print("No data to plot.")
        return

    # ---------- plot ----------
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12
    n = len(violin_data)
    positions = list(range(1, n + 1))
    rank_tag = f' ({ranks})' if ranks else ''

    # 1. Unit conversion: 5.08 cm = 2.0 inches
    target_inch = 5.08 / 2.54

    # 2. Margins for axis labels and ticks (in inches), can be adjusted based on font
    left_margin = 0.4    # left margin for y-axis label
    right_margin = 0.1   # right margin
    bottom_margin = 0.3  # bottom margin for x-axis labels
    top_margin = 0.1     # top margin

    # 3. Calculate final figure size
    fig_w = target_inch + left_margin + right_margin
    fig_h = target_inch + bottom_margin + top_margin

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    parts = ax.violinplot(
        [d for d, _, _, _ in violin_data],
        positions=positions,
        showmeans=False, showmedians=False,
        widths=0.4
    )
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(violin_data[i][2])
        pc.set_alpha(0.7)
    for partname in ('cbars', 'cmins', 'cmaxes'):
        if partname in parts:
            parts[partname].set_visible(False)

    y_tops = []
    for (data, _, color, stars), pos in zip(violin_data, positions):
        if jitter:
            jitter_x = pos + np.random.uniform(-0.1, 0.1, size=len(data))
            ax.scatter(jitter_x, data, color='gray', zorder=2, edgecolors='none', s=np.pi*3)
        q25, q75 = np.percentile(data, 25), np.percentile(data, 75)
        ax.plot([pos, pos], [q25, q75], color='black', linewidth=4, zorder=3)
        ax.plot([pos-0.08, pos+0.08], [np.median(data), np.median(data)],
                color='white', linewidth=2.5, zorder=4)
        ax.scatter([pos], [np.mean(data)], color='white', s=np.pi*3,
                zorder=5, edgecolors='black', linewidths=1.44/2)
        y_tops.append(np.max(data))

    # ---------- Significance markers ----------
    if n_permutations > 0 and any(s for _, _, _, s in violin_data):
        all_vals = np.concatenate([d for d, _, _, _ in violin_data])
        y_range  = np.max(all_vals) - np.min(all_vals)
        offset   = max(y_range * 0.08, 0.05)
        for (data, _, color, stars), pos, y_top in zip(violin_data, positions, y_tops):
            if stars:
                ax.text(pos, y_top + offset * 0.3, stars,
                        ha='center', va='bottom', size=16)
        ax.set_ylim(top=max(y_tops) + offset * 1.6)

    ax.set_xticks(positions)
    ax.set_xticklabels([label for _, label, _, _ in violin_data])
    ax.tick_params(axis='x', bottom=False, top=False, labelbottom=True)
    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(-window_before_stats, window_after_stats)
    ax.set_ylabel(ylabel, size=12)
    # ax.set_title(title)
    ax.axhline(y=0, color='black', linestyle=':', linewidth=1.44)
    ax.spines['bottom'].set_linewidth(1.44)
    ax.spines['left'].set_linewidth(1.44)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', width=1.44)
    # ax.tick_params(axis='x', bottom=False, top=False, labelbottom=True, rotation=45)
    ax.tick_params(axis='y', length=3.06*2)
    # plt.tight_layout()
    if save_fig_dir:
        plt.savefig(f'{os.path.join(save_fig_dir, title)}.eps', dpi=300, bbox_inches='tight')
    plt.show()


def plot_location_around_behavior_event(
    behavior_data,          # Behavior data dictionary
    coordinate_data,        # Coordinate data
    data_type='ret',        # 'ret' or 'lst'
    behavior_name='freezing',  # Behavior name (for the title)
    event_type='onset',     # 'onset' or 'offset'
    if_partner=True,        # True: plot the partner's position; False: plot your own position
    window_before=10,       # Time window before the event (seconds)
    window_after=10,        # Time window after the event (seconds)
    fps=30,                 # Frame rate
    bin_size_frames=1,      # Bin size (in frames)
    # ret-specific parameters
    ret_arena_params=None,  # dict with 'dx', 'rat_x', 'near_width', 'middle_width', 'far_width'
    rat_in_frames=None,     # ret time-correction parameters
    offset_frames=None,     # ret time-correction parameters
    # lst-specific parameters
    lst_arena_params=None,  # dict with 'pixpergrid', 'arena_size', 'edge_width', 'nest_coord', 'nest_size'
    looming_start_frames=None,  # lst looming start frames
    mp_ids=None,            # lst mp_ids
    lstp_data=None,         # lst raw data
    # plot parameters
    figsize=(16, 5),
    dpi=100
):
    """
    Plot the position distribution of the subject or partner within 10 seconds before and after a behavior event (onset/offset)
    
    Parameters:
    -----------
    behavior_data : dict
        Behavior data format: {'p': {'D': {t_id: [[start, end], ...]}, 'S': {...}}}
    coordinate_data : dict or None
        ret: coordinate data obtained from ret_extract_target_data
        lst: set to None to use lstp_data
    data_type : str
        'ret' 或 'lst'
    behavior_name : str
        Behavior name used for the title
    event_type : str
        'onset' (行为开始) 或 'offset' (行为结束)
    if_partner : bool
        True: plot the partner's position; False: plot the subject's position
    window_before : float
        Time window before the event (seconds)
    window_after : float
        Time window after the event (seconds)
    fps : int
        Frame rate
    bin_size_frames : int
        Number of frames in each bin
    
    ret-specific parameters:
    ret_arena_params : dict
        Contains 'dx', 'rat_x', 'near_width', 'middle_width', 'far_width'
    rat_in_frames : dict
        Dictionary of rat entry frame numbers
    offset_frames : dict
        Dictionary of time offset values
        
    lst-specific parameters:
    lst_arena_params : dict
        Contains 'pixpergrid', 'arena_size', 'edge_width', 'nest_coord', 'nest_size'
    looming_start_frames : dict
        Looming start frame for each trial
    mp_ids : list
        List of mp_ids for the batch
    lstp_data : dict
        lst raw coordinate data
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure object
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 计算帧数
    window_frames_before = window_before * fps
    window_frames_after = window_after * fps
    n_bins = (window_frames_before + window_frames_after) // bin_size_frames
    
    # Define the zone-classification function
    if data_type == 'ret':
        # ret: near/middle/far zones
        if ret_arena_params is None:
            raise ValueError("ret_arena_params is required for data_type='ret'")
        
        dx = ret_arena_params['dx']
        rat_x = ret_arena_params['rat_x']
        near_width = ret_arena_params['near_width']
        middle_width = ret_arena_params['middle_width']
        
        near_zone_range = (rat_x, rat_x + near_width * dx)
        middle_zone_range = (rat_x + near_width * dx, rat_x + (near_width + middle_width) * dx)
        far_zone_range = (rat_x + (near_width + middle_width) * dx, rat_x + ret_arena_params.get('arena_size', 24 * dx))
        
        def get_zone(x_coord, y_coord=None):
            """ret: determine the zone from the X coordinate"""
            if np.isnan(x_coord) or (isinstance(x_coord, str) and x_coord == '-'):
                return 'unknown'
            x_coord = float(x_coord)
            if x_coord < near_zone_range[1]:
                return 'near'
            elif middle_zone_range[0] <= x_coord < middle_zone_range[1]:
                return 'middle'
            elif far_zone_range[0] <= x_coord:
                return 'far'
            else:
                return 'unknown'
        
        zone_names = ['near', 'middle', 'far', 'unknown']
        zone_colors = ['orange', 'blue', 'green', 'gray']
        zone_labels = ['Near zone', 'Middle zone', 'Far zone', 'Unknown']
        
    elif data_type == 'lst':
        # lst: center/edge/nest zones (corresponding to ret's near/middle/far)
        if lst_arena_params is None:
            raise ValueError("lst_arena_params is required for data_type='lst'")
        
        edge_width = lst_arena_params['edge_width']
        arena_size = lst_arena_params['arena_size']
        nest_coord = lst_arena_params['nest_coord']
        nest_size = lst_arena_params['nest_size']
        
        def get_zone(x_coord, y_coord):
            """lst: determine the region from the X and Y coordinates"""
            if np.isnan(x_coord) or np.isnan(y_coord):
                return 'unknown'
            
            # Check the nest area
            if (nest_coord[0] <= x_coord <= nest_coord[0] + nest_size[0] and
                nest_coord[1] <= y_coord <= nest_coord[1] + nest_size[1]):
                return 'nest'
            
            # Check the edge area
            if (x_coord < edge_width or x_coord > arena_size - edge_width or
                y_coord < edge_width or y_coord > arena_size - edge_width):
                return 'edge'
            
            return 'center'
        
        # Reorder to center/edge/nest, corresponding to ret's near/middle/far and matching the colors
        zone_names = ['center', 'edge', 'nest', 'unknown']
        zone_colors = ['orange', 'blue', 'green', 'gray']
        zone_labels = ['Center zone', 'Edge zone', 'Nest zone', 'Unknown']
    
    else:
        raise ValueError(f"data_type must be 'ret' or 'lst', got '{data_type}'")
    
    # Initialize the result dictionary
    zone_counts_per_bin = {'D': {}, 'S': {}}
    for rank in ['D', 'S']:
        for bin_idx in range(n_bins):
            zone_counts_per_bin[rank][bin_idx] = {zone: 0 for zone in zone_names}
    
    # Iterate through each behavior bout
    for rank in ['D', 'S']:
        rank_key = 'M' + rank
        partner_rank = 'S' if rank == 'D' else 'D'
        partner_rank_key = 'M' + partner_rank
        
        for t_id, bouts in behavior_data['p'][rank].items():
            # Filter invalid bouts
            if bouts is None or (isinstance(bouts, float) and np.isnan(bouts)):
                continue
            if not isinstance(bouts, list) or len(bouts) == 0:
                continue
            
            bouts = [b for b in bouts if not np.any(np.isnan(b))]
            if not bouts:
                continue
            
            # Determine which mouse's coordinate data to use based on if_partner
            if data_type == 'ret':
                # ret: retrieve from coordinate_data
                trial_key = t_id[:-2] + 'M1&M2'
                
                # Time correction
                corr_rat_in = rat_in_frames[trial_key] - offset_frames[trial_key]
                
                # Determine the coordinate to use based on if_partner
                if if_partner:
                    # Get the partner's X coordinate
                    x_data = coordinate_data['pair'][partner_rank_key]['withrat'][t_id[:-2]].values
                else:
                    # Get the subject's X coordinate
                    x_data = coordinate_data['pair'][rank_key]['withrat'][t_id[:-2]].values
                
                x_data = np.array([np.nan if (isinstance(x, str) and x == '-') else x for x in x_data], dtype=float)
                y_data = None  # ret does not need a Y coordinate
                
            elif data_type == 'lst':
                # lst: retrieve from lstp_data
                batch_num = t_id.split('B')[1].split('M')[0]
                rank_str = t_id.split('M')[1].split('T')[0]
                trial_num = int(t_id.split('T')[1]) - 1
                
                mp_id = mp_ids[int(batch_num) - 1]
                if mp_id not in looming_start_frames or trial_num >= len(looming_start_frames[mp_id]):
                    continue
                
                looming_start_frame = looming_start_frames[mp_id][trial_num]
                
                # Determine the coordinate to use based on if_partner
                if if_partner:
                    # Determine the partner's m value
                    if batch_num in ['1', '3', '7', '10', '11', '12', '13', '14', '15', '16', '17', '20']:
                        target_m = 2 if rank_str == 'S' else 1
                    else:
                        target_m = 2 if rank_str == 'D' else 1
                    target_rank_str = 'S' if rank == 'D' else 'D'
                else:
                    # Determine the subject's m value
                    if batch_num in ['1', '3', '7', '10', '11', '12', '13', '14', '15', '16', '17', '20']:
                        target_m = 1 if rank_str == 'S' else 2
                    else:
                        target_m = 1 if rank_str == 'D' else 2
                    target_rank_str = rank_str
                
                target_m_id = t_id.split('M')[0] + 'M' + str(target_m) + '(' + target_rank_str + ')'
                
                if target_m_id not in lstp_data:
                    continue
                
                x_data = lstp_data[target_m_id]['coordinate']['waist']['x'].values
                y_data = lstp_data[target_m_id]['coordinate']['waist']['y'].values
                corr_rat_in = 0  # lst does not need additional correction
            
            # Iterate through each bout
            for bout in bouts:
                f_start, f_end = bout
                f_start = int(f_start)
                f_end = int(f_end)
                
                # Determine the event time point
                if event_type == 'onset':
                    event_frame = f_start
                elif event_type == 'offset':
                    event_frame = f_end
                else:
                    raise ValueError(f"event_type must be 'onset' or 'offset', got '{event_type}'")
                
                # Apply time correction
                if data_type == 'ret':
                    corr_event_frame = event_frame + corr_rat_in
                elif data_type == 'lst':
                    corr_event_frame = looming_start_frame + event_frame - 150
                
                # Count the zone distribution in each bin
                for bin_idx in range(n_bins):
                    bin_start = corr_event_frame - window_frames_before + bin_idx * bin_size_frames
                    bin_end = bin_start + bin_size_frames
                    
                    if bin_start < 0:
                        continue
                    if bin_end > len(x_data):
                        bin_end = len(x_data)
                    if bin_start >= len(x_data):
                        continue
                    
                    # Get the coordinates in this bin
                    x_in_bin = x_data[bin_start:bin_end]
                    if data_type == 'lst':
                        y_in_bin = y_data[bin_start:bin_end]
                    
                    # Filter out NaNs
                    if data_type == 'ret':
                        valid_indices = ~np.isnan(x_in_bin)
                        x_in_bin_clean = x_in_bin[valid_indices]
                        
                        for x_coord in x_in_bin_clean:
                            zone = get_zone(x_coord)
                            zone_counts_per_bin[rank][bin_idx][zone] += 1
                    
                    elif data_type == 'lst':
                        valid_indices = ~(np.isnan(x_in_bin) | np.isnan(y_in_bin))
                        x_in_bin_clean = x_in_bin[valid_indices]
                        y_in_bin_clean = y_in_bin[valid_indices]
                        
                        for x_coord, y_coord in zip(x_in_bin_clean, y_in_bin_clean):
                            zone = get_zone(x_coord, y_coord)
                            zone_counts_per_bin[rank][bin_idx][zone] += 1
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    
    bin_duration_s = bin_size_frames / fps
    time_positions = np.array([(i * bin_size_frames / fps) - window_before for i in range(n_bins)])
    
    for idx, rank in enumerate(['D', 'S']):
        ax = axes[idx]
        rank_text = 'Dom' if rank == 'D' else 'Sub'
        partner_rank_text = 'Sub' if rank == 'D' else 'Dom'
        
        # Determine the title text based on if_partner
        if if_partner:
            target_text = f'{partner_rank_text} (partner)'
        else:
            target_text = f'{rank_text} (self)'
        
        # Extract the count for each zone
        zone_counts = {}
        for zone in zone_names:
            zone_counts[zone] = np.array([zone_counts_per_bin[rank][i][zone] for i in range(n_bins)])
        
        # Compute totals and percentages
        total_counts = sum(zone_counts.values())
        zone_pcts = {}
        for zone in zone_names:
            zone_pcts[zone] = np.where(total_counts > 0, (zone_counts[zone] / total_counts) * 100, 0)
        
        # Draw the stacked bar plot
        width = bin_duration_s
        bottom = np.zeros(n_bins)
        
        for zone, color, label in zip(zone_names, zone_colors, zone_labels):
            if zone == 'unknown':  # Unknown is usually not displayed
                continue
            ax.bar(time_positions, zone_pcts[zone], width, bottom=bottom,
                   label=label, color=color, alpha=0.8, edgecolor='none', align='edge')
            bottom += zone_pcts[zone]
        
        # Add the event marker line
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, 
                   label=f'{behavior_name.capitalize()} {event_type}', zorder=10)
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Zone time (%)', fontsize=12)
        ax.set_ylim(0, 100)
        ax.set_title(f'{target_text} location around {rank_text} {behavior_name} {event_type}\n'
                     f'(bin={bin_size_frames} frames={bin_duration_s:.3f}s)', fontsize=12)
        ax.set_xlim(-window_before, window_after)
        ax.legend(fontsize=9)
        ax.grid()
    
    plt.tight_layout()
    plt.show()


def plot_behavior_distribution_around_event(
    reference_behavior_data,     # Reference behavior data (e.g., freezing_data)
    all_behavior_labels_dict,    # Dictionary of behavior labels (ret_bhvr_labels_dict or lst_bhvr_labels_dict)
    data_type='ret',             # 'ret' or 'lst'
    reference_behavior='freezing',  # Reference behavior name (for title)
    event_type='onset',          # 'onset' or 'offset'
    if_partner=True,             # True: analyze partner behavior; False: analyze self behavior
    window_before=10,            # Time window before event (seconds)
    window_after=10,             # Time window after event (seconds)
    fps=30,                      # Frame rate
    bin_size_frames=1,           # Bin size (frames)
    # ret-specific parameters
    ret_all_bhvrs=None,          # RET behavior list (order)
    ret_bhvr_abbrev_dict=None,   # RET behavior abbreviation dictionary
    ret_bhvr_color_dict=None,    # RET behavior color dictionary
    # lst-specific parameters
    lst_all_bhvrs=None,          # LST behavior list (order)
    lst_bhvr_abbrev_dict=None,   # LST behavior abbreviation dictionary
    lst_bhvr_color_dict=None,    # LST behavior color dictionary
    # plot parameters
    excluded_behaviors=None,     # List of behaviors to exclude
    composite_behaviors=None,    # Composite behavior definitions, e.g., [('withdrawal', 'approach_partner'), ('huddling', 'freezing')]
    plot_type='stacked_bar',     # 'stacked_bar' or 'line'
    figsize=(14, 6),
    ylabel='Behavior occurrence (%)',
    xlabel=None,
    grid=True,
    legend=True,
    save_fig_dir=None,
    title=None
):
    """
    Plot the distribution of all behaviors around a reference behavior event (onset/offset) for self or partner, using either a stacked bar plot or a line plot.

    Parameters:
    -----------
    reference_behavior_data : dict
        Reference behavior data in the format: {'p': {'D': {t_id: [[start, end], ...]}, 'S': {...}}}
    all_behavior_labels_dict : dict
        Dictionary of all behavior labels containing all possible behaviors
    data_type : str
        'ret' or 'lst'
    reference_behavior : str
        Reference behavior name used in the title
    event_type : str
        'onset' (behavior onset) or 'offset' (behavior offset)
    if_partner : bool
        True: analyze the partner's behavior; False: analyze the subject's own behavior
    window_before : float
        Time window before the event in seconds
    window_after : float
        Time window after the event in seconds
    fps : int
        Frame rate
    bin_size_frames : int
        Number of frames per bin

    ret-specific parameters:
    ret_all_bhvrs : list
        Ret behavior list (in display order)
    ret_bhvr_abbrev_dict : dict
        Ret behavior abbreviation dictionary
    ret_bhvr_color_dict : dict
        Ret behavior color dictionary

    lst-specific parameters:
    lst_all_bhvrs : list
        Lst behavior list (in display order)
    lst_bhvr_abbrev_dict : dict
        Lst behavior abbreviation dictionary
    lst_bhvr_color_dict : dict
        Lst behavior color dictionary

    excluded_behaviors : list or None
        Behaviors to exclude, e.g. ['rat_in']
    composite_behaviors : list of tuple or None
        Composite behavior definitions, e.g. [('withdrawal', 'approach_partner'), ('huddling', 'freezing')]
        When two behaviors occur simultaneously, they are grouped into a composite behavior.
    plot_type : str
        'stacked_bar' or 'line'
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Parameter validation and setup
    if data_type == 'ret':
        if ret_all_bhvrs is None or ret_bhvr_abbrev_dict is None or ret_bhvr_color_dict is None:
            raise ValueError("ret parameters (ret_all_bhvrs, ret_bhvr_abbrev_dict, ret_bhvr_color_dict) are required for data_type='ret'")
        all_bhvrs_list = ret_all_bhvrs
        bhvr_abbrev_dict = ret_bhvr_abbrev_dict
        bhvr_color_dict = ret_bhvr_color_dict
    elif data_type == 'lst':
        if lst_all_bhvrs is None or lst_bhvr_abbrev_dict is None or lst_bhvr_color_dict is None:
            raise ValueError("lst parameters (lst_all_bhvrs, lst_bhvr_abbrev_dict, lst_bhvr_color_dict) are required for data_type='lst'")
        all_bhvrs_list = lst_all_bhvrs
        bhvr_abbrev_dict = lst_bhvr_abbrev_dict
        bhvr_color_dict = lst_bhvr_color_dict
    else:
        raise ValueError(f"data_type must be 'ret' or 'lst', got '{data_type}'")
    
    if excluded_behaviors is None:
        excluded_behaviors = []
    
    # Compute the number of frames
    lookback_frames = int(window_before * fps)
    lookforward_frames = int(window_after * fps)
    total_frames = lookback_frames + lookforward_frames
    n_bins = total_frames // bin_size_frames

    # Add default colors and abbreviations for composite behaviors
    if composite_behaviors:
        for bhvr_pair in composite_behaviors:
            composite_name = '+'.join(bhvr_pair)
            composite_abbrev = '+'.join([bhvr_abbrev_dict.get(b, b) for b in bhvr_pair])
            bhvr_abbrev_dict[composite_name] = composite_abbrev
            # Set a default color for composite behaviors (purple palette)
            if composite_abbrev not in bhvr_color_dict:
                bhvr_color_dict[composite_abbrev] = '#800080'

    # Iterate over the D and S ranks
    for rank in ['D', 'S']:
        text_rank = 'Dom' if rank == 'D' else 'Sub'
        partner_rank = 'S' if rank == 'D' else 'D'
        
        if if_partner:
            text_target = f'Sub (partner)' if rank == 'D' else f'Dom (partner)'
        else:
            text_target = text_rank + ' (self)'
        
        # Initialize per-bin frame counts for each behavior
        behavior_frame_counts = {}
        n_reference_total = 0

        # Get all available behavior categories
        all_behaviors_found = set()
        for t_id in reference_behavior_data['p'][rank].keys():
            partner_t_id = t_id.replace('MD', 'MS') if rank == 'D' else t_id.replace('MS', 'MD')
            if if_partner:
                target_data = all_behavior_labels_dict['p'][partner_rank].get(partner_t_id, {})
            else:
                target_data = all_behavior_labels_dict['p'][rank].get(t_id, {})
            all_behaviors_found.update(target_data.keys())
        
        # Remove excluded behaviors
        all_behaviors_found = [b for b in all_behaviors_found if b not in excluded_behaviors]

        # Sort according to the predefined order and keep only behaviors that are actually present
        all_behaviors = [b for b in all_bhvrs_list if b in all_behaviors_found]

        # Add composite behavior categories
        if composite_behaviors:
            for bhvr_pair in composite_behaviors:
                if all(b in all_behaviors for b in bhvr_pair):
                    composite_name = '+'.join(bhvr_pair)
                    # Insert after the first single behavior
                    idx = all_behaviors.index(bhvr_pair[0])
                    all_behaviors.insert(idx + 1, composite_name)
        
        print(f'[{rank}] Behaviors to analyze (in order): {all_behaviors}')
        
        # Initialize counting arrays for each behavior
        for behavior in all_behaviors:
            behavior_frame_counts[behavior] = np.zeros(n_bins)

        # Iterate over all trials
        for t_id in reference_behavior_data['p'][rank].keys():
            reference_bouts = reference_behavior_data['p'][rank][t_id]
            
            partner_t_id = t_id.replace('MD', 'MS') if rank == 'D' else t_id.replace('MS', 'MD')
            if if_partner:
                target_data = all_behavior_labels_dict['p'][partner_rank].get(partner_t_id, {})
            else:
                target_data = all_behavior_labels_dict['p'][rank].get(t_id, {})
            
            # Filter out invalid bouts
            reference_bouts = [b for b in reference_bouts if not np.any(np.isnan(b))]

            if not reference_bouts:
                continue

            # Count the total number of reference-behavior bouts
            n_reference_total += len(reference_bouts)

            # Sort by starting frame
            reference_bouts = sorted(reference_bouts, key=lambda x: x[0])

            # For each reference-behavior bout, count other behaviors frame by frame
            for ref_start, ref_end in reference_bouts:
                # Determine the event time point
                if event_type == 'onset':
                    event_frame = ref_start
                elif event_type == 'offset':
                    event_frame = ref_end
                else:
                    raise ValueError(f"event_type must be 'onset' or 'offset', got '{event_type}'")

                window_start = max(0, event_frame - lookback_frames)
                window_end = event_frame + lookforward_frames

                # For each frame in the window, check which behaviors are active
                for frame in range(int(window_start), int(window_end)):
                    # Determine which bin the frame belongs to
                    frame_relative = frame - event_frame
                    frame_absolute = frame_relative + lookback_frames
                    bin_idx = frame_absolute // bin_size_frames

                    # Ensure bin_idx is within a valid range
                    if not (0 <= bin_idx < n_bins):
                        continue

                    # Collect all active behaviors at this frame
                    active_behaviors = set()
                    for behavior in all_behaviors:
                        # Skip composite behaviors and handle them separately later
                        if '+' in behavior:
                            continue

                        bhvr_bouts = target_data.get(behavior, [])
                        bhvr_bouts = [b for b in bhvr_bouts if not np.any(np.isnan(b))]

                        # Check whether the frame falls into any bout
                        for bhvr_start, bhvr_end in bhvr_bouts:
                            if bhvr_start <= frame < bhvr_end:
                                active_behaviors.add(behavior)
                                break

                    # Handle overlapping behaviors: if multiple behaviors occur together, group them into a composite behavior
                    if composite_behaviors:
                        for bhvr_pair in composite_behaviors:
                            if all(b in active_behaviors for b in bhvr_pair):
                                composite_name = '+'.join(bhvr_pair)
                                if composite_name in all_behaviors:
                                    behavior_frame_counts[composite_name][bin_idx] += 1
                                    for b in bhvr_pair:
                                        active_behaviors.discard(b)

                    # Count the remaining behaviors
                    for behavior in active_behaviors:
                        behavior_frame_counts[behavior][bin_idx] += 1
        
        print(f'[{rank}] Total {reference_behavior} bouts analyzed: {n_reference_total}')
        
        # Compute the percentage for each behavior
        total_frames_per_bin = n_reference_total * bin_size_frames
        behavior_percentages = {}

        for behavior in all_behaviors:
            behavior_percentages[behavior] = (behavior_frame_counts[behavior] / total_frames_per_bin) * 100

        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 12

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Prepare the time axis
        bin_centers_frames = (np.arange(n_bins) + 0.5) * bin_size_frames
        time_bins = (bin_centers_frames - lookback_frames) / fps
        bin_width_s = bin_size_frames / fps

        if plot_type == 'stacked_bar':
            # Draw a stacked bar plot
            bottom = np.zeros(n_bins)

            for behavior in all_behaviors:
                # Get the abbreviation and color for the behavior
                abbrev = bhvr_abbrev_dict.get(behavior, behavior)
                color = bhvr_color_dict.get(abbrev, '#808080')  # Default gray
                
                ax.bar(time_bins, behavior_percentages[behavior], 
                       width=bin_width_s, bottom=bottom, 
                       label=f'{behavior} ({abbrev})', color=color, edgecolor='none')
                bottom += behavior_percentages[behavior]
            
            ax.set_ylim(0, 100)
            
        elif plot_type == 'line':
            # Draw a line plot; use different line styles when needed
            # linestyles = ['-', '--', '-.', ':']  # solid, dashed, dash-dot, dotted
            linestyles = ['-']  # solid line

            for idx, behavior in enumerate(all_behaviors):
                # Get the abbreviation and color for the behavior
                abbrev = bhvr_abbrev_dict.get(behavior, behavior)
                color = bhvr_color_dict.get(abbrev, '#808080')  # Default gray

                # Cycle through line styles
                linestyle = linestyles[idx % len(linestyles)]
                
                ax.plot(time_bins, behavior_percentages[behavior], 
                        label=f'{behavior} ({abbrev})', 
                        color=color, linewidth=2, linestyle=linestyle, alpha=0.8)
            
            # Set the y-axis range
            max_pct = max(max(behavior_percentages[b]) for b in all_behaviors) if all_behaviors else 0
            ax.set_ylim(0, 105)  # Show at least up to 5%
        else:
            raise ValueError(f"plot_type must be 'stacked_bar' or 'line', got '{plot_type}'")
        
        ax.axvline(0, color='black', linestyle='--', linewidth=1.44, alpha=0.7, 
                   label=f'{text_rank} {reference_behavior} {event_type}')
        print(xlabel)
        if xlabel is None:
            ax.set_xlabel(f'Time aligned to $\mathbf{{{text_rank}}}$ {reference_behavior} (s)', fontsize=12)
        else:
            ax.set_xlabel(xlabel, fontsize=12)
        ax.tick_params(axis='x', bottom=False, top=False, labelbottom=True)
        # ax.set_ylabel(f'Behavior occurrence\n(% of time per bouts)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        # ax.set_title(f'{text_target} behavior distribution around {text_rank} {reference_behavior} {event_type}\n'
        #              f'(±{window_before}/{window_after}s window, bin={bin_size_frames} frames={bin_width_s:.3f}s, n={n_reference_total} {reference_behavior} bouts)', 
        #              fontsize=14)
        ax.set_xlim(-window_before, window_after)
        if legend:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.spines['bottom'].set_linewidth(1.44)
        ax.spines['left'].set_linewidth(1.44)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', width=1.44)
        ax.tick_params(axis='y', length=3.06*2)

        if grid:
            plt.grid()
        # plt.tight_layout()
        if save_fig_dir and title:
            fig_path = f"{save_fig_dir}/{title}_{rank}.eps"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            
            print(f"figure saved to: {fig_path}\n")
        plt.show()


def check_if_bhvr_reduce_investigation_duration(
    behavior_labels_dict,
    target_behavior,
    event_type='offset',
    t_range=(0, 8900),
    data_type='ret',
    fps=30,
    window_before=10,
    window_after=10,
    window_before_stats=1,
    window_after_stats=1,
    n_permutations=1000,
    n_random_total=1000,
    all_bhvrs=None,
    bhvr_abbrev_dict=None,
    bhvr_color_dict=None,
    excluded_behaviors=None,
    composite_behaviors=None,
    filter_method='replace',
    save_prefix='',
    trial_id_format='rat',   # 'rat' or 'looming'
    investigation_behavior='investigation',
    figsize=(6, 5),
    grid=True,
    legend=True,
    save_fig_dir=None,
    title=None,
):
    """
    Check whether the onset/offset of target_behavior reduces the partner's investigation time.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from analyze_data_utils import filter_dict_data, filter_in_range

    np.random.seed(42)

    def get_partner_trial_id(trial_id, rank, trial_id_format):
        partner_rank = 'S' if rank == 'D' else 'D'

        if trial_id_format == 'rat':
            return trial_id.split('M')[0] + 'M' + partner_rank
        elif trial_id_format == 'looming':
            if 'MD' in trial_id:
                return trial_id.replace('MD', 'MS')
            elif 'MS' in trial_id:
                return trial_id.replace('MS', 'MD')
            else:
                return None
        else:
            raise ValueError(f"unknown trial_id_format: {trial_id_format}")

    def is_nan_only_list(x):
        return (
            isinstance(x, list)
            and len(x) == 1
            and isinstance(x[0], (int, float))
            and np.isnan(x[0])
        )

    def filter_bouts_with_partner_investigation(bhvr_data, inv_data, trial_id_format):
        filtered_data = {}

        for group in bhvr_data:
            filtered_data[group] = {}
            for rank in bhvr_data[group]:
                filtered_data[group][rank] = {}

                if not isinstance(bhvr_data[group][rank], dict):
                    continue

                for trial_id, bouts in bhvr_data[group][rank].items():
                    partner_trial_id = get_partner_trial_id(trial_id, rank, trial_id_format)
                    if partner_trial_id is None:
                        continue

                    partner_rank = 'S' if rank == 'D' else 'D'
                    partner_inv = inv_data.get(group, {}).get(partner_rank, {}).get(partner_trial_id, [])

                    if not isinstance(partner_inv, list):
                        continue
                    if not partner_inv or is_nan_only_list(partner_inv):
                        continue

                    filtered_bouts = []
                    for bout in bouts:
                        if not isinstance(bout, (list, tuple, np.ndarray)) or len(bout) < 2:
                            continue

                        bout_start, bout_end = float(bout[0]), float(bout[1])
                        check_frame = bout_end - 1 if event_type == 'offset' else bout_start - 1

                        in_partner_inv = False
                        for inv_bout in partner_inv:
                            if not isinstance(inv_bout, (list, tuple, np.ndarray)) or len(inv_bout) < 2:
                                continue
                            inv_start, inv_end = float(inv_bout[0]), float(inv_bout[1])
                            if inv_start <= check_frame <= inv_end:
                                in_partner_inv = True
                                break

                        if in_partner_inv:
                            filtered_bouts.append(bout)

                    if filtered_bouts:
                        filtered_data[group][rank][trial_id] = filtered_bouts

        return filtered_data

    def calculate_behavior_time_around_timepoint(timepoint, behavior_bouts, window_before_frames, window_after_frames, fps):
        before_window_start = timepoint - window_before_frames
        before_window_end = timepoint
        behavior_before = 0.0

        for b in behavior_bouts:
            if not isinstance(b, (list, tuple, np.ndarray)) or len(b) < 2:
                continue
            b_start, b_end = float(b[0]), float(b[1])
            overlap_start = max(b_start, before_window_start)
            overlap_end = min(b_end, before_window_end)
            if overlap_start < overlap_end:
                behavior_before += (overlap_end - overlap_start)

        after_window_start = timepoint
        after_window_end = timepoint + window_after_frames
        behavior_after = 0.0

        for b in behavior_bouts:
            if not isinstance(b, (list, tuple, np.ndarray)) or len(b) < 2:
                continue
            b_start, b_end = float(b[0]), float(b[1])
            overlap_start = max(b_start, after_window_start)
            overlap_end = min(b_end, after_window_end)
            if overlap_start < overlap_end:
                behavior_after += (overlap_end - overlap_start)

        return behavior_before / fps, behavior_after / fps

    print(f"\n{'='*60}")
    print(f"analyze how {target_behavior} {event_type} affects partner {investigation_behavior}")
    print(f"{'='*60}\n")

    bhvr_data = filter_dict_data(behavior_labels_dict, target_behavior)
    bhvr_data = filter_in_range(bhvr_data, t_range=t_range, method=filter_method)

    inv_data = filter_dict_data(behavior_labels_dict, investigation_behavior)
    inv_data = filter_in_range(inv_data, t_range=t_range, method=filter_method)

    print(f"1. Filter bouts where partner is in {investigation_behavior} at -1 frame")
    filtered_bhvr_data = filter_bouts_with_partner_investigation(bhvr_data, inv_data, trial_id_format)

    total_filtered_bouts = sum(
        len(bouts)
        for group in filtered_bhvr_data
        for rank in filtered_bhvr_data[group]
        for bouts in filtered_bhvr_data[group][rank].values()
    )
    print(f"There are {total_filtered_bouts} bouts\n")

    print("2. Draw filtered behavior distribution")
    # 1. Unit conversion: 5.08 cm = 2.0 inches
    target_inch = 5.08 / 2.54

    # 2. Reserve margins for axis labels and ticks (in inches); adjust if needed for font size
    left_margin = 0.4    # Left margin for the y-axis label
    right_margin = 0.1   # Right margin
    bottom_margin = 0.3  # Bottom margin for the x-axis label
    top_margin = 0.1     # Top margin

    # 3. Compute the final canvas size
    fig_w = target_inch + left_margin + right_margin
    fig_h = target_inch + bottom_margin + top_margin

    plot_behavior_distribution_around_event(
        reference_behavior_data=filtered_bhvr_data,
        all_behavior_labels_dict=behavior_labels_dict,
        data_type=data_type,
        reference_behavior=target_behavior,
        event_type=event_type,
        if_partner=True,
        window_before=window_before,
        window_after=window_after,
        fps=fps,
        bin_size_frames=1,
        **{f'{data_type}_all_bhvrs': all_bhvrs} if all_bhvrs else {},
        **{f'{data_type}_bhvr_abbrev_dict': bhvr_abbrev_dict} if bhvr_abbrev_dict else {},
        **{f'{data_type}_bhvr_color_dict': bhvr_color_dict} if bhvr_color_dict else {},
        excluded_behaviors=excluded_behaviors or [],
        composite_behaviors=composite_behaviors,
        figsize=(fig_w, fig_h),
        xlabel='Time (s)',
        ylabel='Investigation occupancy (%)',
        grid=grid,
        legend=legend,
        save_fig_dir=save_fig_dir,
        title=title
    )

    print(f"3. Calculate partner {investigation_behavior} time before and after event (real)")
    window_before_frames = int(window_before_stats * fps)
    window_after_frames = int(window_after_stats * fps)

    real_data = []
    for group in filtered_bhvr_data:
        for rank in filtered_bhvr_data[group]:
            partner_rank = 'S' if rank == 'D' else 'D'

            for trial_id, bouts in filtered_bhvr_data[group][rank].items():
                partner_trial_id = get_partner_trial_id(trial_id, rank, trial_id_format)
                if partner_trial_id is None:
                    continue

                partner_inv = inv_data.get(group, {}).get(partner_rank, {}).get(partner_trial_id, [])
                if not isinstance(partner_inv, list) or (not partner_inv) or is_nan_only_list(partner_inv):
                    continue

                for bout in bouts:
                    if not isinstance(bout, (list, tuple, np.ndarray)) or len(bout) < 2:
                        continue

                    bout_start, bout_end = float(bout[0]), float(bout[1])
                    timepoint = bout_end if event_type == 'offset' else bout_start

                    before_sec, after_sec = calculate_behavior_time_around_timepoint(
                        timepoint, partner_inv, window_before_frames, window_after_frames, fps
                    )

                    real_data.append({
                        'group': 'real',
                        'trial_id': trial_id,
                        'timepoint': timepoint,
                        'before': before_sec,
                        'after': after_sec,
                        'difference': before_sec - after_sec
                    })

    print(f"There are {len(real_data)} real data points\n")

    print(f"4. Generate {n_random_total} random timepoints (partner also in investigation at -1 frame)")
    trial_info_list = []
    for group in bhvr_data:
        for rank in bhvr_data[group]:
            if not isinstance(bhvr_data[group][rank], dict):
                continue

            partner_rank = 'S' if rank == 'D' else 'D'
            for trial_id in bhvr_data[group][rank].keys():
                partner_trial_id = get_partner_trial_id(trial_id, rank, trial_id_format)
                if partner_trial_id is None:
                    continue

                partner_inv = inv_data.get(group, {}).get(partner_rank, {}).get(partner_trial_id, [])
                if not isinstance(partner_inv, list) or (not partner_inv) or is_nan_only_list(partner_inv):
                    continue

                trial_info_list.append({
                    'group': group,
                    'rank': rank,
                    'trial_id': trial_id,
                    'partner_trial_id': partner_trial_id,
                    'partner_inv': partner_inv
                })

    random_timepoints_data = {}
    random_data = []

    if len(trial_info_list) == 0:
        print("Warning: no valid trial for random generation")
    else:
        generated_count = 0
        max_total_attempts = n_random_total * 100
        attempt_count = 0

        while generated_count < n_random_total and attempt_count < max_total_attempts:
            attempt_count += 1
            trial_info = trial_info_list[np.random.randint(0, len(trial_info_list))]

            random_timepoint = int(np.random.randint(t_range[0], t_range[1]))
            check_frame = random_timepoint - 1

            in_partner_inv = False
            for inv_bout in trial_info['partner_inv']:
                if not isinstance(inv_bout, (list, tuple, np.ndarray)) or len(inv_bout) < 2:
                    continue
                inv_start, inv_end = float(inv_bout[0]), float(inv_bout[1])
                if inv_start <= check_frame <= inv_end:
                    in_partner_inv = True
                    break

            if in_partner_inv:
                before_sec, after_sec = calculate_behavior_time_around_timepoint(
                    random_timepoint, trial_info['partner_inv'], window_before_frames, window_after_frames, fps
                )

                random_data.append({
                    'group': 'random',
                    'trial_id': trial_info['trial_id'],
                    'timepoint': random_timepoint,
                    'before': before_sec,
                    'after': after_sec,
                    'difference': before_sec - after_sec
                })

                g = trial_info['group']
                r = trial_info['rank']
                t = trial_info['trial_id']

                if g not in random_timepoints_data:
                    random_timepoints_data[g] = {}
                if r not in random_timepoints_data[g]:
                    random_timepoints_data[g][r] = {}
                if t not in random_timepoints_data[g][r]:
                    random_timepoints_data[g][r][t] = []

                random_timepoints_data[g][r][t].append((int(random_timepoint - 30), int(random_timepoint)))
                generated_count += 1

        print(f"Generated {len(random_data)} random timepoints after {attempt_count} attempts\n")

    print(f"6. Perform permutation test ({n_permutations} permutations)")

    df_real = pd.DataFrame(real_data)
    df_random = pd.DataFrame(random_data)

    if df_real.empty or df_random.empty:
        print("Not enough data for permutation test. Please check filters/behavior labels.")
        return {
            'real_data_count': len(df_real),
            'random_data_count': len(df_random),
            'df_real': df_real,
            'df_random': df_random
        }

    real_diffs = df_real['difference'].values
    random_diffs = df_random['difference'].values

    observed_mean_diff = np.mean(real_diffs) - np.mean(random_diffs)
    observed_median_diff = np.median(real_diffs) - np.median(random_diffs)

    n_real = len(real_diffs)
    pooled_data = np.concatenate([real_diffs, random_diffs])

    permuted_mean_diffs = []
    permuted_median_diffs = []
    for _ in range(n_permutations):
        shuffled = np.random.permutation(pooled_data)
        perm_real = shuffled[:n_real]
        perm_random = shuffled[n_real:]
        permuted_mean_diffs.append(np.mean(perm_real) - np.mean(perm_random))
        permuted_median_diffs.append(np.median(perm_real) - np.median(perm_random))

    permuted_mean_diffs = np.array(permuted_mean_diffs)
    permuted_median_diffs = np.array(permuted_median_diffs)

    p_value_mean = (np.sum(permuted_mean_diffs >= observed_mean_diff) + 1) / (n_permutations + 1)
    p_value_median = (np.sum(permuted_median_diffs >= observed_median_diff) + 1) / (n_permutations + 1)

    print(f"{'='*30}")
    print(f"real ±{window_before_stats}s mean {investigation_behavior} reduction: {np.mean(real_diffs):.4f} sec")
    print(f"random ±{window_before_stats}s mean {investigation_behavior} reduction: {np.mean(random_diffs):.4f} sec")
    print(f"observed mean difference: {observed_mean_diff:.4f} sec")
    print(f"p-value for mean difference: {p_value_mean:.4f}")
    print()
    print(f"real median reduction: {np.median(real_diffs):.4f} sec")
    print(f"random median reduction: {np.median(random_diffs):.4f} sec")
    print(f"observed median difference: {observed_median_diff:.4f} sec")
    print(f"p-value for median difference: {p_value_median:.4f}")

    print("8. Visualize results")
    fig, axes = plt.subplots(2, 1, figsize=(5, 8))

    ax1 = axes[0]
    parts = ax1.violinplot(
        [real_diffs, random_diffs],
        positions=[1, 2],
        showmeans=False,
        showmedians=False,
        widths=0.4
    )
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(['orange', 'blue'][i])
        pc.set_alpha(0.7)

    for partname in ('cbars', 'cmins', 'cmaxes'):
        if partname in parts:
            parts[partname].set_visible(False)

    for data, pos in [(real_diffs, 1), (random_diffs, 2)]:
        q25 = np.percentile(data, 25)
        q75 = np.percentile(data, 75)
        median = np.median(data)
        mean = np.mean(data)
        ax1.plot([pos, pos], [q25, q75], color='black', linewidth=4, zorder=3)
        ax1.plot([pos - 0.08, pos + 0.08], [median, median], color='white', linewidth=2.5, zorder=4)
        ax1.scatter([pos], [mean], color='white', s=50, zorder=5, edgecolors='black', linewidths=1)

    ax1.set_xticks([1, 2])
    ax1.set_xticklabels([f"Real {target_behavior.upper()}", "Random"])
    ax1.set_xlim(0.5, 2.5)
    ax1.set_ylabel(f"{investigation_behavior} Reduction (sec)", fontsize=12)
    ax1.set_title("Distribution Comparison", fontsize=13)
    ax1.grid(alpha=0.3, axis='y')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    ax2 = axes[1]
    ax2.hist(permuted_mean_diffs, bins=50, alpha=0.7, color='gray', edgecolor='black')
    ax2.axvline(
        observed_mean_diff,
        color='red',
        linestyle='--',
        linewidth=2.5,
        label=f"Observed: {observed_mean_diff:.4f}s"
    )
    ax2.set_xlabel("Mean Difference (Real - Random, sec)", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.set_title(f"Permutation Distribution\np = {p_value_mean:.4f}", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    if save_prefix:
        print("9. Saving results")
        df_combined = pd.concat([df_real, df_random], ignore_index=True)
        data_path = f"{save_prefix}_{target_behavior}_{event_type}_{investigation_behavior}_data.csv"
        df_combined.to_csv(data_path, index=False)

        stats_summary = pd.DataFrame({
            'metric': ['Mean', 'Median', 'Std', 'N'],
            'real': [np.mean(real_diffs), np.median(real_diffs), np.std(real_diffs), len(real_diffs)],
            'random': [np.mean(random_diffs), np.median(random_diffs), np.std(random_diffs), len(random_diffs)],
            'difference': [observed_mean_diff, observed_median_diff, np.nan, np.nan],
            'p_value': [p_value_mean, p_value_median, np.nan, np.nan]
        })
        stats_path = f"{save_prefix}_{target_behavior}_{event_type}_{investigation_behavior}_stats.csv"
        stats_summary.to_csv(stats_path, index=False)

        print(f"data saved to: {data_path}")
        print(f"statistics saved to: {stats_path}\n")

