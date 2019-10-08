import numpy as np
import glob
import os
import csv
from read_audio import AudioProcessor


# def generate_labels():
#     """
#     give labels for multiple classification
#     :return:
#     """
#     label_1 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # >= 5
#     label_2 = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # odd or even
#     label_3 = [0, 0, 1, 1, 1, 1, 1, 0, 0, 0]  # >= 3 <= 7
#     label_4 = [1, 1, 0, 0, 1, 1, 0, 0, 0, 1]  # randomly
#
#     label = np.vstack([label_1, label_2, label_3, label_4]).transpose()
#
#     return label


# def load_sigle_data(path, frame_per_second=100, feature_length=20):
#     """
#     load data from a file
#     :param path:
#     :param frame_per_second:
#     :return:
#     """
#     AP = AudioProcessor(frame_per_second, feature_length, path)
#     # origin = AP.audio_data
#     features = AP.get_feature()
#     if features == []:
#         return []
#     return features

NUM = 10

def save_file(data, save_dir, fname='data.csv'):
    path = os.path.join(save_dir, fname)
    # if not os.path.exists(path):
    #     os.mkdir(path)
    

def data_loader(data_dir, frame_per_second=100, feature_length=20):
    data_base = []
    for idx in range(NUM):
        data_path = os.path.join(data_dir, '*', '{}.*'.format(idx))
        file_list = glob.glob(data_path)
        features_list = []
        for file_path in file_list:
            A = AudioProcessor(frame_per_second, feature_length, file_path)
            features = A.get_feature()
            if features == [] or len(features) < 6 * feature_length:
                print("extract error occurred in {}".format(file_path))
                continue
            features_list.append(features)
            print("Loaded feature {}".format(file_path))
        data_base.append(features_list)
        print("Loaded feature {}".format(idx))
    return data_base


if __name__ == "__main__":
    np.random.seed(5)
    data_dir = "C:\\Users\\wsy\\Desktop\\data_set_z71"
    save_dir = "C:\\Users\\wsy\\Desktop"
    data_base = data_loader(data_dir)
    save_file(data_base, save_dir)
    print(len(data_base))
    print(len(data_base[5]))

