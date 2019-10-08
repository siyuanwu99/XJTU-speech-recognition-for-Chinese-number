import librosa as lb
import librosa.display as lbdis
import matplotlib.pyplot as plt
import os
import numpy as np

FRAME_PER_SECOND = 100  # 25ms per frame


class AudioProcessor:

    def __init__(self, frame_per_second, feature_length: int, path):
        self.frame_per_second = frame_per_second
        self.audio_data, self.sr = lb.load(path, sr=None)
        self.num_origin = len(self.audio_data)  # 总样本数
        self.num_per_frame = int(self.sr / self.frame_per_second)  # 每帧的样本数，窗长
        self.kernel_size = self.num_per_frame
        self.stride = int((self.sr - self.num_per_frame + 1) // self.frame_per_second)
        self.num_frame = int((self.num_origin - self.num_per_frame + 1) // self.stride)  # 剪后总帧数
        self.feature_length = int(feature_length)
        self.eps = 1e-5

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

    def get_avg_zero_rate(self, data, kernel):
        """
        获得短时平均过零率
        :param data:
        :param kernel:
        :return:
        """
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

    def get_boundary_for_multiple_imput(self, data, avg_zero, energy, low_gate=0.08, high_gate=0.25, lmda=0.8):
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
        high_boundary = self._coalesce_boundary_for_multiple(high_boundary, strict=False)
        low_boundary = self._coalesce_boundary_for_multiple(low_boundary)

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

    def get_boundary(self, data, avg_zero, energy, low_gate=0.08, high_gate=0.25, lmda=0.8):
        """
        获取边界/裁剪数据
        :param data: windowed data of audio
        :param avg_zero: average_zero_rate of audio
        :param energy: energy of audio
        :param low_gate:
        :param high_gate:
        :param lmda: select real gate between low gate and high gate
        :return:
        """
        metric = np.max(energy)  # + np.mean(energy)
        high = (energy > high_gate * metric) * 1
        low = 1 - (energy < low_gate * metric)
        diff_high = np.diff(high)
        diff_low = np.diff(low)
        if (np.max(np.abs(diff_high)) <= self.eps) or (np.max(np.abs(diff_low)) <= self.eps):
            return [], []
        if np.min(diff_high) > -1:  # 没录完
            diff_high[-1] = -1
        if np.min(diff_low) > -1:  # 没录完
            diff_low[-1] = -1
        high_boundary = np.vstack([np.where(diff_high == 1), np.where(diff_high == -1)]).transpose()
        low_boundary = np.vstack([np.where(diff_low == 1), np.where(diff_low == -1)]).transpose()
        if (len(high_boundary) > 1) or (len(low_boundary) > 1):
            high_boundary = self._coalesce_boundary(high_boundary, strict=False)
            low_boundary = self._coalesce_boundary(low_boundary)

        boundary = (high_boundary * (1 - lmda) + lmda * low_boundary).astype(np.int).squeeze()
        cropped_data = data[boundary[0]: boundary[1] + 1]
        return cropped_data, boundary

    def _coalesce_boundary_for_multiple(self, boundary, min_length = 20, strict=True):
        """
        合并距离较近的边界
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

    def _coalesce_boundary(self, boundary, min_length = 20, strict=True):
        """
        合并边界
        :param boundary:
        :param min_length:
        :return:
        """
        boundary_ = np.array([boundary[0][0], boundary[-1][-1]])
        return boundary_

    def sum_per_frame_(self):
        """
        summery value to frames without stride
        :return:
        """

        num_frame = int(self.num_origin // self.num_per_frame )
        audio_data = self.audio_data[:num_frame * self.num_per_frame]
        audio_data = np.resize(audio_data, [num_frame, self.num_per_frame])

        print(np.shape(audio_data))
        new_audio_data = np.sum(np.abs(audio_data), axis=1)
        return new_audio_data

    def get_feature(self):
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


if __name__ == "__main__":
    # path = "C:\\Users\\wsy\\Documents\\Audio\\录音 (5).m4a"
    path = "C:\\Users\\wsy\\Desktop\\data_set_z71\\3\\2.wav"
    # audio, sr = lb.load(path, sr=None)
    AP = AudioProcessor(FRAME_PER_SECOND, 20, path)
    origin = AP.audio_data
    kernel = AP.get_window(method='square')
    audio = AP._conv1D(kernel, origin)
    avg_zero_rate = AP.get_avg_zero_rate(origin, kernel)
    energy = AP.get_energy(origin, kernel)
    crp_data, boundary = AP.get_boundary(audio, avg_zero_rate, energy)

    features = AP.get_feature()

    # visualize
    plt.figure(1)
    lbdis.waveplot(audio, sr=FRAME_PER_SECOND)
    plt.title('windowed')

    plt.figure(2)
    lbdis.waveplot(origin, sr=48000)
    plt.title("origin")

    plt.figure(3)
    lbdis.waveplot(avg_zero_rate, sr=FRAME_PER_SECOND)
    plt.title('azr')

    plt.figure(4)
    lbdis.waveplot(energy, sr=FRAME_PER_SECOND)
    plt.title('energy')
    plt.show()

    # plt.figure(5)
    # plt.subplot(1, 3, 1)
    # lbdis.waveplot(crp_data[0], sr=FRAME_PER_SECOND)
    #
    # plt.subplot(1, 3, 2)
    # lbdis.waveplot(crp_data[1], sr=FRAME_PER_SECOND)
    #
    # plt.subplot(1, 3, 3)
    lbdis.waveplot(crp_data, sr=FRAME_PER_SECOND)
    plt.title('crped')

    plt.show()
    print(len(audio))
    print("number of features", len(features))
    print(features[0])
    features = np.array(features)
    print(features.shape)

