import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import json
import pandas as pd
from utils import find_nearest
from config.UAV_60_radar import radar_config
from utils.mappings import labelmap2ra
from config.rudet import radar_configs
import ast

release_dataset_label_map = {0: 'uav'}


def convert(seq_path, single_id, adc_interval_len):
    seq_path = os.path.join(seq_path, f"uav_seqs_{single_id}")
    adc_interval_path = os.path.join(seq_path, f"adc_interval/new_interval_{adc_interval_len}.txt")

    range_grid = labelmap2ra(radar_configs, name='range')
    angle_grid = labelmap2ra(radar_configs, name='angle')

    images_path = os.path.join(seq_path, "camera_to_frame")
    label_path = os.path.join(seq_path, "csv_offset_label")

    files = sorted(os.listdir(label_path))

    file_attributes = 'rtk'
    region_shape_attributes = {"name": "point", "cx": 0, "cy": 0}
    region_attributes = {"class": None}
    columns = ['filename', 'file_size', 'file_attributes', 'region_count', 'region_id',
               'region_shape_attributes', 'region_attributes']
    data = []

    for file in files:
        file_dir = os.path.join(label_path, file)
        label = open(file_dir)
        img_name = file.replace("csv", "jpg")
        img_size = os.path.getsize(os.path.join(images_path, img_name))
        region_count = 0
        obj_info = []

        # parse a label file
        for line in label:
            line = line.rstrip().split(',')
            if int(line[1]) in release_dataset_label_map:
                type_ = release_dataset_label_map[int(line[1])]
            else:
                continue

            angle = float(line[2])
            distance = float(line[3])

            rng_idx, _ = find_nearest(range_grid, distance)
            agl_idx, _ = find_nearest(angle_grid, angle)

            # TODO
            if use_rad:
                obj_info.append([distance, angle, type_])   # RODNet
            else:
                if use_crop:
                    with open(adc_interval_path, "r") as file:
                        content = file.read()
                        new_adc_interval = ast.literal_eval(content.split('\n')[-1])
                    new_rng_idx = int(rng_idx - new_adc_interval[0])
                    assert new_rng_idx >= 0, f"Error."
                    obj_info.append([new_rng_idx, agl_idx, type_])    # RAMP+CropADC
                else:
                    obj_info.append([rng_idx, agl_idx, type_])    # RAMP
            region_count += 1

        for objId, obj in enumerate(obj_info):  # set up rows for different objs
            row = []
            row.append(img_name)
            row.append(img_size)
            row.append(file_attributes)
            row.append(region_count)
            row.append(objId)

            if use_rad:
                region_shape_attributes["cx"] = float(obj[1])  # float --> RODNet
                region_shape_attributes["cy"] = float(obj[0])  # float --> RODNet
            else:
                region_shape_attributes["cx"] = int(obj[1])   # float --> RODNet
                region_shape_attributes["cy"] = int(obj[0])   # float --> RODNet
            if int(obj[0]) == 1:
                print(obj)
            region_attributes["class"] = obj[2]
            row.append(json.dumps(region_shape_attributes))
            row.append(json.dumps(region_attributes))
            data.append(row)

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(os.path.join(os.path.join(seq_path, "annot"), f"ramap_labels_{adc_interval_len}.csv"), index=None, header=True)
    print("\tSuccess!")

    return


if __name__ == "__main__":
    base_root = '/mnt/Data/Share/mmUAV'
    start_single_id = 69
    end_single_id = 69
    adc_interval_len = 256  # 256
    use_rad = False
    use_crop = True


    for single_id in range(start_single_id, end_single_id + 1):
        convert(base_root, single_id, adc_interval_len)
        # res = read_ra_labels_csv(base_root, single_id)
        # print(len(res))

