import ast
import os
import csv
import random

from tqdm import tqdm
from datetime import timedelta
import numpy as np
import re
from geopy.distance import geodesic
import math
import shutil
from lib.instruments import read_mat, visual_ra_rd_ad_from_mat, visual_ra_rd_ad_gt_from_mat
from config.rudet import radar_configs


def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    else:
        os.makedirs(folder_path)


def get_sim_idx(input_key, query_list):
    return np.abs(np.array(query_list) - input_key).argmin()

def deg_to_rad(degrees):
    return degrees * math.pi / 180

def rad_to_deg(rad):
    return rad * 180 / math.pi

def cal_angle(lat1, lon1, lat2, lon2, d_theta):
    lat1, lon1, lat2, lon2 = map(deg_to_rad, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    bearing = math.atan2(x, y)
    bearing = rad_to_deg(bearing)
    bearing = bearing - d_theta
    if bearing <= - 300:
        bearing += 360
    if bearing >= 300:
        bearing -= 360

    assert (bearing < 62 and bearing > - 62) == True, f"error, {print(bearing)}"

    if use_rad:
        return deg_to_rad(bearing)
    else:
        return bearing

def calculate_distance_relative_bearing(lat1, lon1, alt1, lat2, lon2, alt2, d_theta):
    point1 = (lat1, lon1)
    point2 = (lat2, lon2)
    distance = geodesic(point1, point2).meters
    bearing = cal_angle(lat1, lon1, lat2, lon2, d_theta)
    altitude = alt2 - alt1

    return distance, bearing, altitude

def rtk_2_loc(rtkFile, radarStart, radarEnd, labelStart, labelEnd, d_theta):
    lat_lon_dict = {}
    radar_loc = []
    frame_rate = 10

    with open(rtkFile, 'r') as file:
        for line in file:

            match = re.search(r"Timestamp: (.*?), Latitude: (.*?), Longitude: (.*?), Altitude: (.*?)$", line)
            if match:
                timestamp = match.group(1)
                latitude = np.double(match.group(2))
                longitude = np.double(match.group(3))
                altitude = np.double(match.group(4))
                secs = timestamp.split('-')[-1]

  
                if labelStart <= timestamp <= labelEnd:
                    if secs not in lat_lon_dict:
                        lat_lon_dict[secs] = [[latitude, longitude, altitude]]
                    else:
                        lat_lon_dict[secs].append([latitude, longitude, altitude])

         
                if radarStart <= timestamp <= radarEnd:
                    radar_loc.append([latitude, longitude, altitude])


    avg_lat = sum(coord[0] for coord in radar_loc) / len(radar_loc)
    avg_lon = sum(coord[1] for coord in radar_loc) / len(radar_loc)
    avg_alt = sum(coord[2] for coord in radar_loc) / len(radar_loc)

    label_list = []

    for idx in lat_lon_dict:
        sec_list = lat_lon_dict[idx]
        if len(sec_list) < frame_rate:
            sec_list = (sec_list * (frame_rate // len(sec_list) + 1))[:frame_rate]
        else:
            inds = np.linspace(0, len(sec_list) - 1, frame_rate, dtype=int)
            sec_list = [sec_list[i] for i in inds]

        for jdx_ele in sec_list:
            lat2 = jdx_ele[0]
            lon2 = jdx_ele[1]
            alt2 = jdx_ele[2]
            distance, bearing, height = calculate_distance_relative_bearing(avg_lat, avg_lon, avg_alt, lat2, lon2, alt2, d_theta)
            label_list.append([distance, bearing, height])

    return label_list

def annot_generation(base_root, start_seq_id, end_seq_id, frame_num, frame_rate, adc_num, interval_len, rtk_file, viz, add_elevation, get_max_min_idx, adc_crop):

    collect_time = frame_num // frame_rate - 1


    rtk_date = rtk_file.split("/")[-1].split('-')
    y, m, d = rtk_date[0], rtk_date[1], rtk_date[2]

    # TODO: radar position time range
    rS_h, rS_m, rS_s = 22, 14, 20
    rE_h, rE_m, rE_s = 22, 14, 40

    # TODO: Radar azimuth
    d_theta = 323

    
    rS = f"{int(y):04d}-{int(m):02d}-{int(d):02d}_{int(rS_h):02d}-{int(rS_m):02d}-{int(rS_s):02d}"
    rE = f"{int(y):04d}-{int(m):02d}-{int(d):02d}_{int(rE_h):02d}-{int(rE_m):02d}-{int(rE_s):02d}"

    total_rng_ang = []
    for single_id in range(start_seq_id, end_seq_id + 1):
        seq_rng_ang = []
        uav_seqs_dir = base_root + f"/uav_seqs_{single_id}"
        if use_rad:
            csv_save_path = uav_seqs_dir + "/csv_offset_label_rad"
        else:
            csv_save_path = uav_seqs_dir + "/csv_offset_label"

        names_path = uav_seqs_dir + "/camera_to_frame"
        adc_interval_path = uav_seqs_dir + f"/adc_interval/new_interval_{interval_len}.txt"
        rtk_time = os.listdir(os.path.join(uav_seqs_dir, "raw_radar", "azimuth"))[0].split("_")
        lS_h, lS_m, lS_s = rtk_time[0], rtk_time[1], rtk_time[2]

        original_time = timedelta(hours=int(lS_h), minutes=int(lS_m), seconds=int(lS_s))

        updated_time = original_time + timedelta(seconds=collect_time)

        lE_h = updated_time.seconds // 3600
        lE_m = (updated_time.seconds % 3600) // 60
        lE_s = updated_time.seconds % 60


        lS = f"{int(y):04d}-{int(m):02d}-{int(d):02d}_{int(lS_h):02d}-{int(lS_m):02d}-{int(lS_s):02d}"
        lE = f"{int(y):04d}-{int(m):02d}-{int(d):02d}_{int(lE_h):02d}-{int(lE_m):02d}-{int(lE_s):02d}"

        label_list = rtk_2_loc(rtk_file, rS, rE, lS, lE, d_theta)

        # Constructing RA annotation
        clear_folder(csv_save_path)
        for id in tqdm(range(frame_num)):
            csv_data = [100, 0,  label_list[id][1], label_list[id][0], label_list[id][2]]
            file_name = sorted(os.listdir(names_path))[id].split('/')[-1].split('.')[0]

            with open(csv_save_path + f"/{file_name}.csv", 'w') as file:
                writer = csv.writer(file)
                writer.writerow(csv_data)

        if viz:
            from config.UAV_60_radar import radar_config

            draw_label = False

            rng_grid = radar_config['rng_grid'][radar_configs['crop_num']:-radar_configs['crop_num']]
            agl_grid = radar_config['agl_grid']

            start_fid = 0
            end_fid = 299

            azimuth_ra_path = os.path.join(uav_seqs_dir, "python_slice_frame", "azimuth", f'raw_frame_RA')
            azimuth_rd_path = os.path.join(uav_seqs_dir, "python_slice_frame", "azimuth", f'raw_frame_RD')
            azimuth_ad_path = os.path.join(uav_seqs_dir, "python_slice_frame", "azimuth", f'raw_frame_AD')
            elevation_ra_path = os.path.join(uav_seqs_dir, "python_slice_frame", "elevation", f'raw_frame_RA')
            elevation_rd_path = os.path.join(uav_seqs_dir, "python_slice_frame", "elevation", f'raw_frame_RD')
            elevation_ad_path = os.path.join(uav_seqs_dir, "python_slice_frame", "elevation", f'raw_frame_AD')
            viz_save_path = os.path.join(uav_seqs_dir, "visualization")

            for fid in tqdm(range(start_fid, end_fid + 1)):
                azimuth_ra_data = os.path.join(azimuth_ra_path, f'{fid:03d}_{128:09d}.npy')
                azimuth_rd_data = os.path.join(azimuth_rd_path, f'{fid:09d}.npy')
                azimuth_ad_data = os.path.join(azimuth_ad_path, f'{fid:09d}.npy')

                azimuth_ra = np.load(azimuth_ra_data)
                azimuth_rd = np.load(azimuth_rd_data)
                azimuth_ad = np.load(azimuth_ad_data)

                viz_save_data = os.path.join(viz_save_path, f'{fid:09d}.jpg')

                if add_elevation:
                    elevation_ra_data = os.path.join(elevation_ra_path, f'{fid:03d}_{128:09d}.npy')
                    elevation_rd_data = os.path.join(elevation_rd_path, f'{fid:09d}.npy')
                    elevation_ad_data = os.path.join(elevation_ad_path, f'{fid:09d}.npy')

                    elevation_ra = np.load(elevation_ra_data)
                    elevation_rd = np.load(elevation_rd_data)
                    elevation_ad = np.load(elevation_ad_data)
                else:
                    elevation_ra = None
                    elevation_rd = None
                    elevation_ad = None

                if draw_label:
                    rng_ele, agl_ele, alt_ele = label_list[fid]
                    rng_idx = get_sim_idx(rng_ele, rng_grid)
                    agl_idx = get_sim_idx(agl_ele, agl_grid)

                    if adc_crop:
                        with open(adc_interval_path, 'r') as file:
                            new_adc_interval = ast.literal_eval(file.read().split('\n')[-1])
                        rng_idx -= new_adc_interval[0]

                    visual_ra_rd_ad_gt_from_mat(azimuth_ra, azimuth_rd, azimuth_ad, elevation_ra, elevation_rd,
                                                elevation_ad, viz_save_data, [int(rng_idx), agl_idx])
                else:
                    visual_ra_rd_ad_from_mat(azimuth_ra, azimuth_rd, azimuth_ad, elevation_ra, elevation_rd,
                                             elevation_ad, viz_save_data)



        if get_max_min_idx:
            from config.UAV_60_radar import radar_config
            rng_grid = radar_config['rng_grid'][radar_configs['crop_num']:-radar_configs['crop_num']]
            agl_grid = radar_config['agl_grid']
            for fid in range(radar_config['frame_number'] * radar_config['duration']):
                rng_ele, agl_ele, alt_ele = label_list[fid]
                rng_idx = get_sim_idx(rng_ele, rng_grid)
                agl_idx = get_sim_idx(agl_ele, agl_grid)


                seq_rng_ang.append([rng_idx, agl_idx])
                total_rng_ang.append([rng_idx, agl_idx])

            max_a, min_a = max(x[0] for x in seq_rng_ang), min(x[0] for x in seq_rng_ang)
            max_b, min_b = max(x[1] for x in seq_rng_ang), min(x[1] for x in seq_rng_ang)


            new_interval = [0, adc_num - 1]
            if adc_crop is True:
                mid_rng = (max_a + min_a) // 2
                mid_interval_len = interval_len // 2
                interval_left, interval_right = mid_rng-mid_interval_len, mid_rng+mid_interval_len

                if interval_right > adc_num - 1:
                    new_interval = [adc_num - interval_len, adc_num - 1]
                elif interval_left < 0:
                    new_interval = [0, interval_len - 1]
                elif max_a - min_a == interval_len - 1:
                    new_interval = [min_a, max_a]
                else:
                    move_bin = min_a - interval_left
                    assert move_bin >= 0, f"The trimming length must be greater than the sequence length!"
                    random_range_trans = random.randint(-move_bin + 1, move_bin)
                    new_interval = [max(0, interval_left + random_range_trans), max(0, interval_left + random_range_trans) + interval_len - 1]
                    if new_interval[1] > adc_num:
                        new_interval = [new_interval[0] - (new_interval[1] - adc_num + 1), adc_num - 1]
                print(new_interval, min_a, max_a - 1)
                assert new_interval[1] - new_interval[0] == interval_len - 1 and new_interval[0] >= 0 and new_interval[1] <= adc_num
                assert new_interval[0] <= min_a and new_interval[1] >= max_a - 1

                with open(adc_interval_path, "w") as file:
                    file.write(f"Max Range: {max_a}, Min Range: {min_a}, Update the sequences interval to: \n[{new_interval[0]},{new_interval[1]}]")  

            print(f"Seq_id: {single_id}, Max_rng: {max_a}, Min_rng: {min_a}, Max_ang: {max_b}, Min_ang: {min_b}, d_dis:{max_a-min_a}, Interval: {new_interval},  Boundary Length: {min_a - new_interval[0]}.")

    if get_max_min_idx:
        max_a, min_a = max(x[0] for x in total_rng_ang), min(x[0] for x in total_rng_ang)
        max_b, min_b = max(x[1] for x in total_rng_ang), min(x[1] for x in total_rng_ang)
        print(f"All Sequences the Max_rng: {max_a}, Min_rng: {min_a}, Max_ang: {max_b}, Min_ang: {min_b}, d_dis:{max_a-min_a}, Length: {len(total_rng_ang)}.")


if __name__ == '__main__':
    base_root = "/mnt/Data/Share/mmUAV"
    rtk_file = "./RTKDataFile/2025-01-06-RTKData.txt"

    # pay attention
    use_rad = False
    start_seq_id = 1
    end_seq_id = 79
    frame_num = 300
    frame_rate = 10
    adc_num = 512
    interval_len = 512
    viz = True
    add_elevation = False
    get_max_min_idx = True
    adc_crop = True
    random.seed(42)
    annot_generation(base_root, start_seq_id, end_seq_id, frame_num, frame_rate, adc_num, interval_len, rtk_file, viz, add_elevation,
                     get_max_min_idx, adc_crop)
