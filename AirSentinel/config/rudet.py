from config.UAV_60_radar import radar_config


data_sets = {
    'root_dir': "/mnt/Data/Share/mmUAV",
}

train_sets = {
    'root_dir': "/mnt/Data/Share/mmUAV",
    'seqs': [
        1, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 46, 48, 49, 50, 51,
        52, 53, 54, 56, 57, 58, 60, 61, 62, 64, 66, 67, 68, 69, 70, 72, 75, 76, 78, 79
    ],
}

test_type = 1       #{0: total, 1: validset, 2: FB, 3: hovering, 4: LR}
if test_type == 0:
    valid_sets = {
        'root_dir': "/mnt/Data/Share/mmUAV",
        'seqs': [
            60, 61, 62, 64, 66, 67, 68, 69, 70, 72, 75, 76, 78, 79, 2, 6, 7, 8, 37, 45, 47, 55, 59, 63, 65, 71, 73, 74, 77, 1, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 46, 48, 49, 50, 51,
            52, 53, 54, 56, 57, 58
        ],
    }
elif test_type == 1:
    valid_sets = {
        'root_dir': "/mnt/Data/Share/mmUAV",
        'seqs': [
            2, 6, 7, 8, 37, 45, 47, 55, 59, 63, 65, 71, 73, 74, 77
        ],
    }
elif test_type == 2:
    # Forward-Backward Flight
    valid_sets = {
        'root_dir': "/mnt/Data/Share/mmUAV",
        'seqs': [
            2, 55, 71, 73
        ],
    }
elif test_type == 3:
    # Hovering
    valid_sets = {
        'root_dir': "/mnt/Data/Share/mmUAV",
        'seqs': [
            7, 8, 63, 74, 77
        ],
    }
else:
    # Left-Right Flight
    valid_sets = {
        'root_dir': "/mnt/Data/Share/mmUAV",
        'seqs': [
            6, 37, 45, 47, 59, 65
        ],
    }

test_sets = {
    'root_dir': "/mnt/Data/Share/mmUAV",
    'seqs': [
        2, 6, 7, 8, 37, 45, 47, 55, 59, 63, 65, 71, 73, 74, 77
    ],
}


n_class = 1

class_table = {
    0: 'uav'
}

class_ids = {
    'uav': 0
}

confmap_sigmas = {
    'uav': 10
}
confmap_sigmas_interval = {
    'uav': [5, 10]
}

confmap_length = {
    'uav': 1
}

object_sizes = {
    'uav': 0.5
}


camera_configs = {
    'image_width': 1920,
    'image_height': 1080,
    'frame_rate': radar_config['frame_number'],
    'image_folder': 'camera_to_frame',
}


radar_configs = {
    'data_type': 'RISEP',  # 'RI': real + imaginary, 'AP': amplitude + phase
    'rvaemap_rsize': radar_config['ramap_rsize'],  # RVAEMap range size
    'rvaemap_vsize': radar_config['ramap_vsize'],  # RVAEMap velcity size
    'rvaemap_asize': radar_config['ramap_asize'],  # RVAEMap azimth size
    'rvaemap_esize': radar_config['ramap_esize'],  # RVAEMap elevation size
    'frame_rate': radar_config['frame_number'],
    'crop_num': radar_config['num_crop'],
    'n_chirps': radar_config['loop'],
    'sample_freq': radar_config['Fs'],
    'sweep_slope': radar_config['sweepSlope'],
    'ramap_rsize_label': radar_config['ramap_rsize_label'],
    'ramap_asize_label': radar_config['ramap_asize_label'],
    'ra_min_label': radar_config['ra_min_label'],  # min radar angle
    'ra_max_label': radar_config['ra_max_label'],  # max radar angle
    'rr_min': radar_config['rr_min'],              # min radar range (fixed)
    'rr_max': radar_config['rr_max'],              # max radar range (fixed)
    'ra_min': radar_config['ra_min'],              # min radar angle (fixed)
    'ra_max': radar_config['ra_max'],              # max radar angle (fixed)
}


model_configs = {
    'AirSentinel': True,
    'T-RODNet': False,
    'RODNet-CDC': False,
    'RODNet-HG': False,
    'RODNet-HGwI': False,
    'DCSN': False,
    'E-RODNet': False,
    'RAMP': False,
}


# network settings
rudet_configs = {
    #'mnet_cfg': [64, 32],#False
    'IS_Augdata': False,
    'IS_MixAug': False,
    'Norm': True,
    'TVFEM': True,
    'IWCA': True,
    'GTMH': True,
    'n_epoch': 200,
    'train_patience': 50,
    'start_val_epoch': 9,
    'batch_size': 4,
    'guard_bandwidth': 2,
    'out_channels': 16,
    'stacked_num': 1,
    'test_stride': 4,
    'val_stride': 4,
    'val_epoch': 1,
    'learning_rate': 1e-5,
    'lr_step': 5,    # lr will decrease 10 times after lr_step epoches
    'log_step': 50,
    'win_size': 16,
    'r_rate': 4,
    'v_rate': 2,
    'a_rate': 1,
    'input_rsize': 512,
    'input_asize': 128,
    'max_dets': 1,
    'peak_thres': 0.3,
    'ols_thres': 0.3,
    'rr_min': radar_config['rr_min'],  # min radar range
    'rr_max': radar_config['rr_max'],  # max radar range
    'ra_min': radar_config['ra_min'],  # min radar angle
    'ra_max': radar_config['ra_max'],  # max radar angle
    'rr_min_eval': radar_config['rr_min'],  # min radar range
    'rr_max_eval': radar_config['rr_max'],  # max radar range
    'ra_min_eval': radar_config['ra_min'],  # min radar angle
    'ra_max_eval': radar_config['ra_max'],  # max radar angle
}

semi_loss_err_reg = {
    # index unit
    'level1': 30,
}
# correct error region for level 1
err_cor_reg_l1 = {
    'top': 3,
    'bot': 3,
}
# correct error region for level 2
err_cor_reg_l2 = {
    'top': 3,
    'bot': 25,
}
# correct error region for level 3
err_cor_reg_l3 = {
    'top': 3,
    'bot': 35,
}

ra_real_mean = 2.315067046997594e-11
ra_real_std = 1210.6564173652177
ra_imag_mean = 2.7395949205931494e-10
ra_imag_std = 1210.6251044415276

rv_real_mean = -1.500305379718387
rv_real_std = 1791.6382938826605
rv_imag_mean = 0.0753188710520044
rv_imag_std = 1787.4468229488093

va_real_mean = -0.5146794940065592
va_real_std = 369.2562293027373
va_imag_mean = 3.5573185792099684
va_imag_std = 369.28189300414846
