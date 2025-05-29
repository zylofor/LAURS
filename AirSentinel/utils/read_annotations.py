import os
import pandas as pd
import json
from utils import find_nearest
from config.rudet import class_ids
from config.rudet import rudet_configs, radar_configs
from utils.mappings import confmap2ra, labelmap2ra

range_grid = confmap2ra(radar_configs, name='range')
angle_grid = confmap2ra(radar_configs, name='angle')
range_grid_label = labelmap2ra(radar_configs, name='range')
angle_grid_label = labelmap2ra(radar_configs, name='angle')


def read_ra_labels_csv(seq_path, single_id, adc_length):
    seq_path = os.path.join(seq_path, f"uav_seqs_{single_id}")
    label_csv_name = os.path.join(seq_path, "annot", f'ramap_labels_{adc_length}.csv')
    data = pd.read_csv(label_csv_name)
    n_row, n_col = data.shape
    print(n_row)
    obj_info_list = []
    cur_idx = -1

    for r in range(n_row):
        filename = data['filename'][r]
        frame_idx = int(filename.split('.')[0].split('_')[-1])
        if cur_idx == -1:
            obj_info = []
            cur_idx = frame_idx
        if frame_idx > cur_idx:
            obj_info_list.append(obj_info)
            obj_info = []
            cur_idx = frame_idx

        region_count = data['region_count'][r]
        region_id = data['region_id'][r]

        if region_count != 0:
            region_shape_attri = json.loads(data['region_shape_attributes'][r])
            region_attri = json.loads(data['region_attributes'][r])

            cx = region_shape_attri['cx']
            cy = region_shape_attri['cy']
            distance = range_grid_label[cy]
            angle = angle_grid_label[cx]
            # if distance > rudet_configs['rr_max'] or distance < rudet_configs['rr_min']:
            #     continue
            # if angle > rudet_configs['ra_max'] or angle < rudet_configs['ra_min']:
            #     continue
            rng_idx, _ = find_nearest(range_grid, distance)
            agl_idx, _ = find_nearest(angle_grid, angle)

            # rng_idx = region_shape_attri['cx']
            # agl_idx = region_shape_attri['cy']

            try:
                class_str = region_attri['class']
            except:
                print("missing class at row %d" % r)
                continue
            try:
                class_id = class_ids[class_str]
            except:
                if class_str == '':
                    print("no class label provided!")
                    raise ValueError
                else:
                    class_id = -1000
                    print("Warning class not found! %s %010d" % (seq_path, frame_idx))
            obj_info.append([rng_idx, agl_idx, class_id])

    obj_info_list.append(obj_info)

    return obj_info_list
