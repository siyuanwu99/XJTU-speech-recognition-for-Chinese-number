import librosa as lb
import librosa.display as lbdis
import matplotlib.pyplot as plt
import tsfresh.feature_extraction.feature_calculators as feature_calc
import numpy as np
import fft

NUM_PER_FRAME = 128  # 7.8125ms per frame


def norm(input):
    mean = np.mean(input)
    max = np.max(input)
    return (input - mean) / max


class AudioProcessor:

    def __init__(self, num_per_frame, path, local_feature_length: int = 20,
                 mfcc_order: int = 16, mfcc_cof: int = 10):

        self.num_per_frame = num_per_frame  # 每帧的样本数，窗长
        self.meta_audio_data, self.sr = lb.load(path, sr=None)  # sr 是采样率 sample rate
        self.num_origin = len(self.meta_audio_data)  # 总样本数
        self.frame_per_second = int(self.sr / self.num_per_frame)
        self.kernel_size = num_per_frame  # 核的大小，即每帧的长度, 窗长
        self.stride = int(7 / 12 * self.kernel_size)  # 步长
        self.num_frame = int((self.num_origin - self.num_per_frame + 1) // self.stride)  # 剪后总帧数
        self.feature_length = int(local_feature_length)

        self.mfcc_order = mfcc_order  # n
        self.mfcc_cof = mfcc_cof  # m
        self.eps = 1e-5

    def _coalesce_multiple_boundary(self, boundary, min_length=20, strict=True):
        """
        多词情况，合并距离较近的边界
        :param boundary:
        :param min_length:
        :return:
        """
        if boundary.shape[0] < boundary.shape[1]:
            boundary = boundary.transpose()
        length = len(boundary)
        boundary_list = []

        for i in range(length):
            if i < length - 1:
                if boundary[i + 1][0] - boundary[i][1] < min_length:
                    boundary[i][1] = boundary[i + 1][1]
                    boundary[i + 1][0] = boundary[i][0]
                    continue

                if strict & (boundary[i][1] - boundary[i][0] < 0.8 * min_length):
                    continue

                boundary_list.append(boundary[i])
        return boundary_list

    def _coalesce_boundary(self, boundary, min_length=20, strict=True):
        """
        单词情况，合并边界
        :param boundary:
        :param min_length:
        :return:
        """
        boundary_ = np.array([boundary[0][0], boundary[-1][-1]])
        return boundary_

    def _conv1D(self, kernel, data):
        """
        1-dimentional convolution 一维卷积，这是加窗的函数
        :param kernel:卷积核
        :param data:
        :return:
        """
        new_audio = np.zeros(self.num_frame)
        for i in range(self.num_frame):
            new_audio[i] = np.dot(kernel, data[i * self.stride: i * self.stride + self.num_per_frame])
        return new_audio

    def _add_window(self, kernel, data):
        new_audio_list = []
        for i in range(self.num_frame):
            frame = kernel * data[i * self.stride: i * self.stride + self.num_per_frame]
            new_audio_list.append(frame)
        new_audio = np.vstack(new_audio_list)
        assert len(new_audio_list) == self.num_frame
        return new_audio

    def _mfcc_filter(self, num_filters, low_freq, high_freq):
        # low_freq = np.min(freq)
        # high_freq = np.max(freq)

        F_mel = lambda x: 1125 * (np.log(x / 700) + 1)
        F_mel_converse = lambda x: 700 * (np.exp(x / 1125) - 1)

        f_filter = np.zeros(num_filters)
        for m in range(num_filters):
            f_filter[m] = self.kernel_size / self.sr * F_mel_converse(
                F_mel(low_freq) + m * (F_mel(high_freq) - F_mel(low_freq)) / (num_filters + 1)
            )

        H = np.zeros([num_filters, self.kernel_size])
        for m in range(num_filters - 3):
            for k in range(self.kernel_size - 1):
                if (k <= f_filter[m + 1]) and (k >= f_filter[m]):
                    H[m, k] = (k - f_filter[m]) / (f_filter[m + 1] - f_filter[m])
                elif (k <= f_filter[m + 2]) and (k >= f_filter[m + 1]) and (m <= num_filters - 3):
                    H[m, k] = (f_filter[m + 2] - k) / (f_filter[m + 2] - f_filter[m + 1])
                elif k > f_filter[num_filters - 1]:
                    H[m, k] = 1
        # print(np.sum(H, axis=0))
        return H

    def _discrete_cosine_transform(self):
        cos_ary = np.zeros([self.mfcc_cof, self.mfcc_order])  # (M, N)
        for n in range(self.mfcc_order - 1):
            for m in range(self.mfcc_cof - 1):
                cos_ary[m, n] = np.cos(np.pi * n * (2 * m - 1) / 2 / self.mfcc_cof)
        return cos_ary

    def get_window(self, method="hanning"):
        """
        get windows to audio data 语音信号生成窗
        Method should in [square, hanning, hamming]
        :return:
        """
        kernel_size = self.num_per_frame

        if method == 'square':
            kernel = np.ones([1, kernel_size])
        elif method == 'hanning':
            kernel = np.hanning(kernel_size)[np.newaxis, :]
        elif method == 'hamming':
            kernel = np.hamming(kernel_size)[np.newaxis, :]
        else:
            Exception('Undefined method!')
            kernel = np.zeros([1, kernel_size])
        return kernel

    def get_avg_zero_rate(self, data, kernel):
        """
        获得短时平均过零率
        :param data:
        :param kernel:
        :return:
        """
        # for i, x in enumerate(data):
        #     if abs(x) < 1e-4:
        #         data[i] = 0
        tmp_data = np.insert(data[:-1], 0, 0)
        processed_data = np.abs(np.sign(tmp_data) - np.sign(data))
        processed_data[0] = 0
        avg_zero_rate = self._conv1D(kernel, processed_data)
        return avg_zero_rate

    def get_energy(self, data, kernel):
        """
        获得能量
        :param data:
        :param kernel:
        :return:
        """
        kernel = kernel ** 2 / self.kernel_size
        energy_data = self._conv1D(kernel, data ** 2)
        return energy_data

    def get_upper_rate(self, data):
        '''
        梯度大于零部分与全部的比值
        :param data:
        :return:
        '''
        dif = np.diff(data)
        up_zero = np.sum((dif > 0) * 1.0)
        return up_zero / len(data)

    def get_multiple_boundary(self, data, avg_zero, energy, low_gate=0.08, high_gate=0.25, lmda=0.8):
        """
        对同文件多词获取边界/裁剪数据
        :param data: windowed data of audio
        :param avg_zero: average_zero_rate of audio
        :param energy: energy of audio
        :param low_gate:
        :param high_gate:
        :param lmda: select real gate between low gate and high gate
        :return:
        """

        cropped_data = []
        cropped_boundary = []

        metric = np.max(energy)  # + np.mean(energy)
        high = (energy > high_gate * metric) * 1
        low = 1 - (energy < low_gate * metric)
        diff_high = np.diff(high)
        diff_low = np.diff(low)
        high_boundary = np.vstack([np.where(diff_high == 1), np.where(diff_high == -1)]).transpose()
        low_boundary = np.vstack([np.where(diff_low == 1), np.where(diff_low == -1)]).transpose()
        high_boundary = self._coalesce_multiple_boundary(high_boundary, strict=False)
        low_boundary = self._coalesce_multiple_boundary(low_boundary)

        if len(high_boundary) != len(low_boundary):
            print("Error, this file can't be loaded "
                  "because high bound and low bound didn't match")
            return cropped_data, cropped_boundary

        assert len(high_boundary) == len(low_boundary)
        assert len(high_boundary) <= 10

        for i in range(len(high_boundary)):
            if not np.cumprod(high_boundary[i] - low_boundary[i])[1] < 0:  # lb and hb not match
                pass
            boundary = (high_boundary[i] * (1 - lmda) + lmda * low_boundary[i]).astype(
                np.int)  # using the average between high and low
            # boundary = low_boundary[i]
            cropped_data.append(data[boundary[0]: boundary[1] + 1])
            cropped_boundary.append(boundary)
        return cropped_data, cropped_boundary

    def get_boundary(self, input, low_gate=0.05, high_gate=0.25, lmda=0.8):
        """
        获取边界/裁剪数据
        :param data: windowed data of audio
        :param avg_zero: average_zero_rate of audio
        :param input: energy of audio
        :param low_gate:
        :param high_gate:
        :param lmda: select real gate between low gate and high gate
        :return:
        """
        metric = np.max(input)  # + np.mean(input)
        high = (input > high_gate * metric) * 1
        low = 1 - (input < low_gate * metric)
        if high[0] > 0 or high[-1] > 0:  # 预防没有录全的情况
            high[0] = 0
            high[-1] = 0
        if low[0] > 0 or low[-1] > 0:
            low[0] = 0
            low[-1] = 0
        diff_high = np.diff(high)
        diff_low = np.diff(low)
        if (np.max(np.abs(diff_high)) <= self.eps) or (np.max(np.abs(diff_low)) <= self.eps):
            return [], []  # 没有录上声音

        high_boundary = np.vstack([np.where(diff_high == 1), np.where(diff_high == -1)]).transpose()
        low_boundary = np.vstack([np.where(diff_low == 1), np.where(diff_low == -1)]).transpose()
        if (len(high_boundary) > 1) or (len(low_boundary) > 1):
            high_boundary = self._coalesce_boundary(high_boundary, strict=False)
            low_boundary = self._coalesce_boundary(low_boundary)

        boundary = (high_boundary * (1 - lmda) + lmda * low_boundary).astype(np.int).squeeze()
        return boundary

    def get_local_feature(self):
        """
        Only extracted local features from given signals.
        This function is not useful in this experiment.
        :return:
        """
        square_kernel = self.get_window(method='square')
        hanning_kernel = self.get_window(method='hanning')
        square_data = self._conv1D(square_kernel, self.meta_audio_data)
        hanning_data = self._conv1D(hanning_kernel, self.meta_audio_data)
        square_azrate = self.get_avg_zero_rate(self.meta_audio_data, square_kernel)
        hanning_azrate = self.get_avg_zero_rate(self.meta_audio_data, hanning_kernel)
        square_energy = self.get_energy(self.meta_audio_data, square_kernel)
        hanning_energy = self.get_energy(self.meta_audio_data, hanning_kernel)

        # cut
        crp_data, crp_boundary = self.get_boundary(square_energy)

        # extract feature
        lb = crp_boundary[0]
        if (crp_data == []) | (crp_boundary == []):
            return []
        square_feature = np.hstack([square_data[lb: lb + self.feature_length],
                                    square_azrate[lb: lb + self.feature_length],
                                    square_energy[lb: lb + self.feature_length]])
        hanning_feature = np.hstack([hanning_data[lb: lb + self.feature_length],
                                     hanning_azrate[lb: lb + self.feature_length],
                                     hanning_energy[lb: lb + self.feature_length]])
        feature = np.hstack([square_feature, hanning_feature])
        return feature

    def get_global_feature(self):
        """
        获取时域全局特征，包含最大值、标准差、平均值
        :param hadcropped:
        :return:
        """
        square_data, square_energy, square_azrate = self.pre_process(method='hanning', ifcrop=True)
        func = lambda x: [
            # feature_calc.autocorrelation(norm(x), 5),
            np.std(x),
            feature_calc.approximate_entropy(norm(x), 5, 1),
            feature_calc.cid_ce(x, normalize=True),
            feature_calc.count_above_mean(x),
            feature_calc.first_location_of_minimum(x),
            feature_calc.first_location_of_maximum(x),
            feature_calc.last_location_of_maximum(x),
            feature_calc.last_location_of_minimum(x),
            feature_calc.longest_strike_above_mean(x),
            feature_calc.number_crossing_m(x, 0.8*np.max(x)),
            feature_calc.skewness(x),
            feature_calc.time_reversal_asymmetry_statistic(x, 5)
                          ]
        # global features I want to get
        upper_rate = self.get_upper_rate(square_energy)
        feature = np.hstack([
            [np.mean(norm(square_energy))],
            [upper_rate],
            func(square_azrate),
            func(square_energy)
        ])
        return feature

    def get_mfcc_feature(self, hadcropped=False):
        '''
        calculate Mel-frequency cepstral coefficients in frequency domain and extract features from MFCC
        :return: numpy array
        '''
        assert self.frame_per_second not in [32, 64, 128, 256], \
            Exception("Cannot operate butterfly computation ,"
                      "frame per second should in [32, 64, 128, 256]")
        hanning_kernel = self.get_window(method='hanning')
        windowed = self._add_window(hanning_kernel, self.meta_audio_data)  # [num_frame, kernel_size]
        hanning_energy = self.get_energy(self.meta_audio_data, hanning_kernel)

        if not hadcropped:
            boundary = self.get_boundary(hanning_energy)
            cropped = windowed[boundary[0]: boundary[1] + 1, :]
            frequency = np.vstack([fft.fft(frame.squeeze()) for frame in np.vsplit(cropped, len(cropped))])
        else:
            frequency = np.vstack([fft.fft(windowed)])
        frequency = np.abs(frequency)
        frequency_energy = frequency ** 2

        low_freq = self.sr / self.num_per_frame
        high_freq = self.sr

        H = self._mfcc_filter(self.mfcc_cof, low_freq, high_freq)
        S = np.dot(frequency_energy, H.transpose())  # (F, M)
        cos_ary = self._discrete_cosine_transform()
        mfcc_raw_features = np.sqrt(2 / self.mfcc_cof) * np.dot(S, cos_ary)  # （F，N)

        upper = [self.get_upper_rate(fea) for fea in mfcc_raw_features.transpose()]
        assert len(upper) == mfcc_raw_features.shape[1]

        func = lambda x: [
            # feature_calc.autocorrelation(norm(x), 5),
            np.std(x),
            feature_calc.approximate_entropy(norm(x), 5, 1),
            feature_calc.cid_ce(x, normalize=True),
            feature_calc.count_above_mean(x),
            feature_calc.first_location_of_minimum(x),
            feature_calc.first_location_of_maximum(x),
            feature_calc.last_location_of_maximum(x),
            feature_calc.last_location_of_minimum(x),
            feature_calc.longest_strike_above_mean(x),
            feature_calc.number_crossing_m(x, 0.8*np.max(x)),
            feature_calc.skewness(x),
            feature_calc.time_reversal_asymmetry_statistic(x, 5)
                          ]

        mfcc_features = np.hstack(
            [func(col) for col in mfcc_raw_features.transpose()]

        )
        return mfcc_features

    def get_combined_feature(self, hadcropped):
        '''
        Get combined time domain and frequency domain features.
        :return: numpy array
        '''
        time_domain = self.get_global_feature(hadcropped=hadcropped)
        freq_domain = self.get_mfcc_feature(hadcropped=hadcropped)
        return np.hstack([time_domain, freq_domain])

    def sum_per_frame_(self):
        """
        简单的加和， stride == kernel 的卷积。
        已弃用
        :return:
        """

        num_frame = int(self.num_origin // self.num_per_frame)
        audio_data = self.meta_audio_data[:num_frame * self.num_per_frame]
        audio_data = np.resize(audio_data, [num_frame, self.num_per_frame])

        print(np.shape(audio_data))
        new_audio_data = np.sum(np.abs(audio_data), axis=1)
        return new_audio_data

    def pre_process(self, method, ifcrop=True):
        """
        预处理数据，返回绝对平均值、过零率、能量
        :param method: window method
        :param ifcrop:
        :return: average, average zero rate, energy
        """
        kernel = self.get_window(method=method)
        avg_data = self._conv1D(kernel, self.meta_audio_data)
        azrate = self.get_avg_zero_rate(self.meta_audio_data, kernel)
        energy = self.get_energy(self.meta_audio_data, kernel)

        if ifcrop:
            boundary = self.get_boundary(energy)
            avg_data = avg_data[boundary[0]: boundary[1] + 1]
            azrate = azrate[boundary[0]: boundary[1] + 1]
            energy = energy[boundary[0]: boundary[1] + 1]
        return avg_data, azrate, energy


if __name__ == "__main__":
    # load files
    # path = "C:\\User\\wsy\\Desktop\\录音 (3).m4a"
    path1 = "C:\\Users\\wsy\\Desktop\\dataset3\\lgt\\9\\data5.wav"
    # path2 = "C:\\Users\\wsy\\Desktop\\dataset3\\lgt\\8\\data10.wav"
    # path3 = "C:\\Users\\wsy\\Desktop\\dataset3\\lgt\\3\\data13.wav"

    AP1 = AudioProcessor(num_per_frame=NUM_PER_FRAME, path=path1)
    # AP2 = AudioProcessor(num_per_frame=NUM_PER_FRAME, path=path2)
    # AP3 = AudioProcessor(num_per_frame=NUM_PER_FRAME, path=path3)

    origin1 = AP1.meta_audio_data
    avg1, azr1, eng1 = AP1.pre_process(method='hanning')
    feature = AP1.get_global_feature()
    print(feature)

    # origin2 = AP2.meta_audio_data
    # avg2, azr2, eng2 = AP2.pre_process(method='hanning')
    #
    # origin3 = AP3.meta_audio_data
    # avg3, azr3, eng3 = AP3.pre_process(method='hanning')

    # visualize
    # plt.figure(1)
    # lbdis.waveplot(eng1, sr=NUM_PER_FRAME)
    # plt.title('Number 9 lgt')
    # plt.savefig("C:\\Users\\wsy\\Desktop\\lgt_9.png")
    # plt.show()
    #
    # plt.figure(2)
    # lbdis.waveplot(origin2, sr=NUM_PER_FRAME)
    # plt.title('Number 8 lgt')
    # plt.savefig("C:\\Users\\wsy\\Desktop\\wsy_8.png")
    # plt.show()
    #
    # plt.figure(2)
    # lbdis.waveplot(origin3, sr=NUM_PER_FRAME)
    # plt.title('Number 3 lgt')
    # plt.savefig("C:\\Users\\wsy\\Desktop\\lst_3.png")
    # plt.show()


