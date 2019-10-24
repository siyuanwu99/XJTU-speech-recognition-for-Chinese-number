import librosa as lb
import librosa.display as lbdis
import matplotlib.pyplot as plt
import os
import numpy as np
import fft

FRAME_PER_SECOND = 100  # 25ms per frame


def norm(input):
    mean = np.mean(input)
    std = np.std(input)
    return (input - mean) / std


class AudioProcessor:

    def __init__(self, num_per_frame, path, feature_length: int = 20,
                 mfcc_order: int = 16, mfcc_cof: int = 10):

        self.num_per_frame = num_per_frame  # 每帧的样本数，窗长
        self.audio_data, self.sr = lb.load(path, sr=None)  # sr 是采样率 sample rate
        self.num_origin = len(self.audio_data)  # 总样本数
        self.frame_per_second = int(self.sr / self.num_per_frame)
        self.kernel_size = num_per_frame  # 核的大小，即每帧的长度, 窗长
        self.stride = int(7 / 12 * self.kernel_size)  # 步长
        self.num_frame = int((self.num_origin - self.num_per_frame + 1) // self.stride)  # 剪后总帧数
        self.feature_length = int(feature_length)

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

    def get_window(self, method="square"):
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
            boundary = (high_boundary[i] * (1 - lmda) + lmda * low_boundary[i]).astype(np.int)   # using the average between high and low
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
        get feature from audio with ten numbers
        # TODO: this function only include features from 'square' and 'hanning' windows
        :return:
        """
        square_kernel = self.get_window(method='square')
        hanning_kernel = self.get_window(method='hanning')
        square_data = self._conv1D(square_kernel, self.audio_data)
        hanning_data = self._conv1D(hanning_kernel, self.audio_data)
        square_azrate = self.get_avg_zero_rate(self.audio_data, square_kernel)
        hanning_azrate = self.get_avg_zero_rate(self.audio_data, hanning_kernel)
        square_energy = self.get_energy(self.audio_data, square_kernel)
        hanning_energy = self.get_energy(self.audio_data, hanning_kernel)

        # cut
        crp_data, crp_boundary = self.get_boundary(square_data, square_azrate, square_energy)

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

    def get_global_feature(self, iscropped=False):
        square_kernel = self.get_window(method='square')
        hanning_kernel = self.get_window(method='hanning')
        square_data = self._conv1D(square_kernel, np.abs(self.audio_data))
        hanning_data = self._conv1D(hanning_kernel, np.abs(self.audio_data))
        square_azrate = self.get_avg_zero_rate(self.audio_data, square_kernel)
        hanning_azrate = self.get_avg_zero_rate(self.audio_data, hanning_kernel)
        square_energy = self.get_energy(self.audio_data, square_kernel)
        hanning_energy = self.get_energy(self.audio_data, hanning_kernel)

        # crop if haven't been cropped
        if not iscropped:
            boundary = self.get_boundary(square_energy)
            square_energy = square_energy[boundary[0]: boundary[1]+1]
            hanning_energy = hanning_energy[boundary[0]: boundary[1] + 1]
            square_data = square_data[boundary[0]: boundary[1] + 1]
            hanning_data = hanning_data[boundary[0]: boundary[1] + 1]
            square_azrate = square_azrate[boundary[0]: boundary[1] + 1]
            hanning_azrate = hanning_azrate[boundary[0]: boundary[1] + 1]

        func = lambda x: [np.max(x), np.std(x), np.mean(x)]
        upper_rate = self.get_upper_rate(square_energy)
        feature = np.hstack([
            func(square_energy),
            func(hanning_energy),
            func(square_azrate),
            func(hanning_azrate),
            func(square_data),
            func(hanning_data),
            [upper_rate]
        ])
        return feature

    def get_mfcc_feature(self):
        '''

        :return: numpy array
        '''
        assert self.frame_per_second not in [32, 64, 128, 256], \
            Exception("Cannot operate butterfly computation ,"
                      "frame per second should in [32, 64, 128, 256]")
        hanning_kernel = self.get_window(method='hanning')
        windowed = self._add_window(hanning_kernel, self.audio_data)  # [num_frame, kernel_size]
        hanning_energy = self.get_energy(self.audio_data, hanning_kernel)
        boundary = self.get_boundary(hanning_energy)
        cropped = windowed[boundary[0] : boundary[1] + 1, :]
        frequency = np.vstack([fft.fft(frame.squeeze()) for frame in np.vsplit(cropped, len(cropped))])
        frequency = np.real(frequency)  # TODO: real or mode?
        frequency_energy = frequency ** 2

        low_freq = self.sr / self.num_per_frame
        high_freq = self.sr

        H = self._mfcc_filter(self.mfcc_cof, low_freq, high_freq)
        S = np.dot(frequency_energy, H.transpose())  # (F, M)
        cos_ary = self._discrete_cosine_transform()
        mfcc_raw_features = np.sqrt(2 / self.mfcc_cof) * np.dot(S, cos_ary)  #（F，N)
        mfcc_features = np.hstack(
            [np.sum(mfcc_raw_features, axis=0),
             np.max(mfcc_raw_features, axis=0),
             np.min(mfcc_raw_features, axis=0),
             np.std(mfcc_raw_features, axis=0)
            ]
        )
        return mfcc_features

    def sum_per_frame_(self):
        """
        简单的加和， stride == kernel 的卷积
        :return:
        """

        num_frame = int(self.num_origin // self.num_per_frame )
        audio_data = self.audio_data[:num_frame * self.num_per_frame]
        audio_data = np.resize(audio_data, [num_frame, self.num_per_frame])

        print(np.shape(audio_data))
        new_audio_data = np.sum(np.abs(audio_data), axis=1)
        return new_audio_data


if __name__ == "__main__":
    # path = "C:\\Users\\wsy\\Documents\\Audio\\录音 (5).m4a"
    path = "C:\\Users\\wsy\\Desktop\\dataset3\\lgt\\4\\data2.wav"
    # audio, sr = lb.load(path, sr=None)
    AP = AudioProcessor(feature_length=30,
                        frame_per_second=FRAME_PER_SECOND,
                        path=path)
    origin = AP.audio_data
    kernel = AP.get_window(method='square')
    audio = AP._conv1D(kernel, origin)
    avg_zero_rate = AP.get_avg_zero_rate(origin, kernel)
    energy = AP.get_energy(origin, kernel)
    boundary = AP.get_boundary(energy)

    features = AP.get_global_feature()

    # visualize
    plt.figure(1)
    lbdis.waveplot(audio, sr=FRAME_PER_SECOND)
    plt.title('windowed')
    plt.show()

    plt.figure(2)
    lbdis.waveplot(origin, sr=48000)
    plt.title("origin")
    plt.show()

    plt.figure(3)
    lbdis.waveplot(avg_zero_rate, sr=FRAME_PER_SECOND)
    plt.title('azr')
    plt.show()

    plt.figure(4)
    lbdis.waveplot(energy, sr=FRAME_PER_SECOND)
    plt.title('energy')
    plt.show()

    plt.figure(5)
    lbdis.waveplot(audio[boundary[0]:boundary[1]+1], sr=FRAME_PER_SECOND)
    plt.title('cropped_avg')
    plt.show()

    plt.figure(6)
    lbdis.waveplot(energy[boundary[0]:boundary[1]+1], sr=FRAME_PER_SECOND)
    plt.title('cropped energy')
    plt.show()

    print(len(audio))
    print("number of features", len(features))
    print(features[0])
    features = np.array(features)
    print(features.shape)
