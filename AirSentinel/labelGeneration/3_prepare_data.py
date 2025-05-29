import os
import numpy as np
import pickle
import argparse
from tqdm import tqdm
from utils.confidence_map import generate_confmap, normalize_confmap, add_noise_channel
from utils.read_annotations import read_ra_labels_csv
from utils.visualization import visualize_confmap

from config.rudet import train_sets, test_sets, valid_sets
from config.rudet import n_class, radar_configs, rudet_configs


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare RUDNet data.')
    parser.add_argument('-m', '--mode', type=str, dest='mode', help='choose from train, valid, test, supertest')
    parser.add_argument('-dd', '--datadir', type=str, dest='data_dir', default='/mnt/Data/Share/mmUAV/train_test_seqs',
                        help='data directory to save the prepared data')
    args = parser.parse_args()
    return args


def prepare_data(sets, set_type='train', viz=False):
    base_root = '/mnt/Data/Share/mmUAV'
    adc_length = 128
    sets_seqs = sets['seqs']

    seqs = sets_seqs
    for seq in tqdm(seqs):
        detail_list = [[], 0]
        confmap_list = [[], []]
        n_data = 300
        # create paths for data
        for fid in range(n_data):
            if radar_configs['data_type'] == 'RISEP' or radar_configs['data_type'] == 'APSEP':
                path = os.path.join(base_root, f"uav_seqs_{seq}", "python_slice_frame", "azimuth", chirp_folder_name, "%03d_%09d.npy")
            else:
                raise ValueError
            detail_list[0].append(path)

        # use labelled RAMap
        try:
            obj_info_list = read_ra_labels_csv(base_root, seq, adc_length)
        except Exception as e:
            print("Load sequence %s failed!" % base_root)
            print(e)
            continue
        assert len(obj_info_list) == n_data

        for obj_info in obj_info_list:
            confmap_gt = np.zeros((n_class + 1, int(radar_configs['rvaemap_rsize'] // rudet_configs['r_rate']), int(radar_configs['rvaemap_asize'] // rudet_configs['a_rate'])),
                                  dtype=float)
            confmap_gt[-1, :, :] = 1.0
            if len(obj_info) != 0:
                confmap_gt = generate_confmap(obj_info)
                confmap_gt = normalize_confmap(confmap_gt)
                confmap_gt = add_noise_channel(confmap_gt)
            assert confmap_gt.shape == (n_class + 1, int(radar_configs['rvaemap_rsize'] // rudet_configs['r_rate']), int(radar_configs['rvaemap_asize'] // rudet_configs['a_rate']))
            if viz:
                visualize_confmap(confmap_gt)
            confmap_list[0].append(confmap_gt)
            confmap_list[1].append(obj_info)
            # end objects loop
        confmap_list[0] = np.array(confmap_list[0])

        dir2 = os.path.join(base_root, "train_test_seqs", "confmaps_gt", set_type)
        dir3 = os.path.join(base_root, "train_test_seqs", "data_details", set_type)
        if not os.path.exists(dir2):
            os.makedirs(dir2)
        if not os.path.exists(dir3):
            os.makedirs(dir3)

        # save pkl files (每个pkl存放一个序列的confmap和obj_info)
        pickle.dump(confmap_list, open(os.path.join(dir2, f"uav_seqs_{seq}" + '_azimuth.pkl'), 'wb'))
        # save pkl files
        pickle.dump(detail_list, open(os.path.join(dir3, f"uav_seqs_{seq}" + '_azimuth.pkl'), 'wb'))

        # end frames loop
    # end seqs loop
# end dates loop


if __name__ == "__main__":
    """
    Example:
        python 3_prepare_data.py -m train -dd './data/'
    """
    args = parse_args()
    modes = args.mode.split(',')
    data_dir = args.data_dir

    if radar_configs['data_type'] == 'RI':
        chirp_folder_name = 'radar_chirps_RI'
    elif radar_configs['data_type'] == 'AP':
        chirp_folder_name = 'radar_chirps_AP'
    elif radar_configs['data_type'] == 'RISEP':
        chirp_folder_name = 'raw_frame_RA'
    elif radar_configs['data_type'] == 'APSEP':
        chirp_folder_name = 'radar_chirps_win_APSEP'
    else:
        raise ValueError

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)


    for mode in modes:
        if mode == 'train':
            print('Preparing %s sets ...' % mode)
            prepare_data(train_sets, set_type=mode, viz=False)
        elif mode == 'valid':
            print('Preparing %s sets ...' % mode)
            prepare_data(valid_sets, set_type=mode, viz=False)
        elif mode == 'test':
            print('Preparing %s sets ...' % mode)
            prepare_data(test_sets, set_type=mode, viz=False)
        else:
            print("Warning: unknown mode %s" % mode)
