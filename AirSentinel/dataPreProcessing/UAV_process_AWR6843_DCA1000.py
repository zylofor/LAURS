import os
import numpy as np
from scipy.signal.windows import taylor
from tqdm import tqdm
import ast

# 定义雷达对象参数
class RadarObject():
    def __init__(self):
        # 路径配置
        self.root = "/mnt/Data/Share/mmUAV"
        self.start_single_id = 1
        self.end_single_id = 79
        self.create_type = ['Azimuth'] # 'Azimuth', 'Elevation'

        # 6843 -- 配置
        self.numTX = 3
        self.numRX = 4
        self.numADCSamples = 512
        self.idxProcChirp = 255
        self.numLanes = 2
        self.framePerSecond = 10
        self.duration = 30

        # FFT -- 配置
        self.rangeBins = 512
        self.dopplerBins = 128
        self.azimuthBins = 128
        self.crop_num = 12
        self.rangeBins = self.rangeBins + 2 * self.crop_num
        self.use_filter = True
        self.use_adc_interval = True
        self.adc_interval_length = 128

        # 定义文件保存列表
        self.radarDataFileNameGroup = []
        self.saveDirNameGroup = []
        self.rgbFileNameGroup = []
        self.jointsFileNameGroup = []
        self.initialize(self.start_single_id, self.end_single_id)


    def initialize(self, start_single_id, end_single_id):
        for i in range(start_single_id, end_single_id + 1):
            radarDataFileName = [os.path.join(self.root, f"uav_seqs_{i}", "raw_radar", "azimuth"), os.path.join(self.root, f"uav_seqs_{i}", "raw_radar", "elevation")]
            if self.use_filter:
                saveDirName = os.path.join(self.root, f"uav_seqs_{i}", "python_slice_frame")
            else:
                saveDirName = os.path.join(self.root, f"uav_seqs_{i}", "python_slice_frame_0")
            self.check_path(saveDirName)
            self.radarDataFileNameGroup.append(radarDataFileName)
            self.saveDirNameGroup.append(saveDirName)


    def check_same(self, x, y):
        x = np.array(x).reshape(-1)
        y = np.array(y).reshape(-1)
        return np.array_equal(x, y)


    def check_path(self, x):
        if type(x) is str:
            x = [x]
        for x_i in x:
            if os.path.exists(x_i) is False:
                os.mkdir(x_i)

    def getAdcDataFromDCA1000(self, fileName):
        adcDataName = sorted(os.listdir(fileName))
        adcData = []
        for i, adc_data in enumerate(adcDataName):
            adc_data_tmp = np.fromfile(os.path.join(fileName, adcDataName[i]), dtype=np.int16)
            adcData.append(adc_data_tmp)
        adcData = np.concatenate(adcData)
        adcData = np.transpose(adcData.reshape(-1, 2, 2), axes=(0, 2, 1)).reshape(-1, 2)
        adcData = adcData[:, 0] + (np.sqrt(-1 + 0j) * adcData[:, 1])
        adcData = np.transpose(adcData.reshape(-1, self.numRX, self.numADCSamples), axes=(1, 0, 2))
        return adcData

    def clutterRemoval(self, input_val, axis=0):
        reordering = np.arange(len(input_val.shape))
        reordering[0] = axis
        reordering[axis] = 0
        input_val = input_val.transpose(reordering)

        # Apply static clutter removal
        mean = input_val.transpose(reordering).mean(0)
        output_val = input_val - np.expand_dims(mean, axis=0)
        out = output_val.transpose(reordering)
        return out


    def generateHeatmap(self, frame):
        data_first = frame[:, 0::3, :]
        data_second = frame[:, 1::3, :]
        data_third = frame[:, 2::3, :]
        # Azimuth-Doppler-Range
        data_merge = np.concatenate([data_first, data_second, data_third], axis=0)
        if self.use_filter:
            data_merge = data_merge - np.mean(data_merge, axis=1, keepdims=True)

        # Range-FFT
        range_win = np.hamming(self.numADCSamples)
        merge_range = np.fft.fft(data_merge * range_win[np.newaxis, np.newaxis, :], axis=2, n=self.rangeBins)

        # create RD
        merge_RD = np.fft.fft(merge_range, axis=1, n=self.dopplerBins)
        merge_RD = np.fft.fftshift(merge_RD, axes=1)
        RD = np.mean(merge_RD[:, :, self.crop_num:-self.crop_num], axis=0, keepdims=False)
        RD = np.transpose(RD, axes=[1, 0])
        RD = np.concatenate([np.expand_dims(np.real(RD), axis=-1), np.expand_dims(np.imag(RD), axis=-1)], axis=-1)

        # create RA
        azimuth_win = taylor(self.numRX * self.numTX)
        RA = np.fft.fft(merge_range * azimuth_win[:, np.newaxis, np.newaxis], axis=0, n=self.azimuthBins)
        RA = np.fft.fftshift(RA, axes=0)
        RA = RA[:, :, self.crop_num:-self.crop_num]
        RA = np.transpose(RA, axes=[1, 2, 0])
        RA = np.concatenate([np.expand_dims(np.real(RA), axis=-1), np.expand_dims(np.imag(RA), axis=-1)], axis=-1)

        # create AD
        AD = np.fft.fft(merge_RD * azimuth_win[:, np.newaxis, np.newaxis], axis=0, n=self.azimuthBins)
        AD = np.fft.fftshift(AD, axes=0)
        AD = np.mean(AD[:, :, self.crop_num:-self.crop_num], axis=2, keepdims=False)
        AD = np.concatenate([np.expand_dims(np.real(AD), axis=-1), np.expand_dims(np.imag(AD), axis=-1)], axis=-1)

        return RA, RD, AD

    def saveRadarData(self, save_data, save_path, idxFrame, adc_interval_path=None):
        AD_save_path = os.path.join(save_path, "raw_frame_AD")
        RA_save_path = os.path.join(save_path, "raw_frame_RA")
        RD_save_path = os.path.join(save_path, "raw_frame_RD")
        self.check_path([save_path, AD_save_path, RA_save_path, RD_save_path])
        RA, RD, AD = save_data
        RA = RA.astype(np.float32)
        RD = RD.astype(np.float32)
        AD = AD.astype(np.float32)

        if adc_interval_path is not None:
            with open(adc_interval_path, "r") as file:
                content = file.read()
                adc_interval = ast.literal_eval(content.split('\n')[-1])
            RD = RD[adc_interval[0]: adc_interval[1] + 1, ...]

        np.save(os.path.join(AD_save_path, f"{idxFrame:09d}.npy"), AD)
        np.save(os.path.join(RD_save_path, f"{idxFrame:09d}.npy"), RD)

        for RA_idx in range(RA.shape[0]):
            # chirps Range Azimuth 2
            RA_item = RA[RA_idx, ...]

            if adc_interval_path is not None:
                RA_item = RA_item[adc_interval[0]: adc_interval[1] + 1, ...]
            np.save(os.path.join(RA_save_path, f"{idxFrame:03d}_{RA_idx+1:09d}.npy"), RA_item)


    def processRadarDataAzimuth(self):
        for idxName in range(len(self.radarDataFileNameGroup)):
            if 'Azimuth' in self.create_type:
                # TX all Range
                adcDataAzimuths = self.getAdcDataFromDCA1000(self.radarDataFileNameGroup[idxName][0])
                adcDataAzimuths = adcDataAzimuths.reshape(self.numRX, -1, self.numTX * self.idxProcChirp, self.numADCSamples)
                adcDataAzimuths = np.split(adcDataAzimuths, adcDataAzimuths.shape[1], axis=1)
                if self.use_adc_interval:
                    adc_interval_path = os.path.join(self.root, self.radarDataFileNameGroup[idxName][0].split("/")[-3], f"adc_interval/new_interval_{self.adc_interval_length}.txt")
                else:
                    adc_interval_path = None
                for idxFrame in range(len(adcDataAzimuths)):
                    frameAzimuth = adcDataAzimuths[idxFrame].squeeze()
                    outputAzimuth = self.generateHeatmap(frameAzimuth)
                    self.saveRadarData(outputAzimuth, os.path.join(self.saveDirNameGroup[idxName], 'azimuth'), idxFrame, adc_interval_path)
                    print(f"Saving the azimuth sequence {idxName}： {idxFrame} frame.")

            if 'Elevation' in self.create_type:
                adcDataAzimuths = self.getAdcDataFromDCA1000(self.radarDataFileNameGroup[idxName][1])
                adcDataAzimuths = adcDataAzimuths.reshape(self.numRX, -1, self.numTX * self.idxProcChirp, self.numADCSamples)
                adcDataAzimuths = np.split(adcDataAzimuths, adcDataAzimuths.shape[1], axis=1)

                for idxFrame in tqdm(range(len(adcDataAzimuths))):
                    frameAzimuth = adcDataAzimuths[idxFrame].squeeze()
                    outputAzimuth = self.generateHeatmap(frameAzimuth)
                    self.saveRadarData(outputAzimuth, os.path.join(self.saveDirNameGroup[idxName], 'elevation'), idxFrame)
                    print(f"Saving the elevation sequence {idxName}： {idxFrame} frame.")

if __name__ == '__main__':
    radarObject = RadarObject()
    radarObject.processRadarDataAzimuth()





