import numpy as np
import glob
import os
import json
from audio_processor import AudioProcessor
from utils import save_data


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


def save_file(data, save_dir, fname='data.json'):
    """
    ERROR Json file can't not be saved
    :param data:
    :param save_dir:
    :param fname:
    :return:
    """
    path = os.path.join(save_dir, fname)
    js_data = json.dumps(data)
    with open(path, 'w+', encoding='UTF-8') as file:
        file.write(js_data)
    print("Extracted data saved in {}".format(path))


def load_file(save_path):
    """
    ERROR: json file can't be saved and loaded
    :param save_path:
    :return:
    """
    with open(save_path, 'r+', encoding='UTF-8') as file:
        js = file.read()
        data = json.load(js)
    return data


def data_loader(data_dir, frame_per_second=100, feature_length=20):
    """
    load all data from data_dir
    :param data_dir:
    :param frame_per_second:
    :param feature_length:
    :return: list of array
    """
    data_set = []
    for idx in range(NUM):

        # find file with label idx

        # for data dir label -> person
        data_path = os.path.join(data_dir, '*', '{}'.format(idx), '*.wav')
        # for data dir person -> label
        # data_path = os.path.join(data_dir, str(idx), '*')

        file_list = glob.glob(data_path)
        features_list = []

        # get audio data from file
        for file_path in file_list:
            A = AudioProcessor(frame_per_second, feature_length, file_path)
            features = A.get_global_feature(iscropped=False)
            if features == []:
                print("extract error occurred in {}".format(file_path))
                continue
            features_list.append(features)
            print("Loaded feature {}".format(file_path))
        number = len(features_list)
        features = np.vstack(features_list)
        labels = np.ones([number, 1]) * idx
        data = np.hstack([features, labels])
        data_set.append(data)
        print("Loaded feature {}".format(idx))
    return data_set


def mfcc_loader(data_dir, frame_per_second, mfcc_cof, mfcc_ord):
    """
    load all data from data_dir
    :param data_dir:
    :param frame_per_second:
    :param feature_length:
    :return: list of array
    """
    data_set = []
    for idx in range(NUM):

        # find file with label idx

        # for data dir label -> person
        data_path = os.path.join(data_dir, '*', '{}'.format(idx), '*.wav')
        # for data dir person -> label
        # data_path = os.path.join(data_dir, str(idx), '*')

        file_list = glob.glob(data_path)
        features_list = []

        # get audio data from file
        for file_path in file_list:
            A = AudioProcessor(frame_per_second, file_path,
                               mfcc_cof=mfcc_cof, mfcc_order=mfcc_ord)
            features = A.get_mfcc_feature()
            if features == []:
                print("extract error occurred in {}".format(file_path))
                continue
            features_list.append(features)
            print("Loaded feature {}".format(file_path))
        number = len(features_list)
        features = np.vstack(features_list)
        labels = np.ones([number, 1]) * idx
        data = np.hstack([features, labels])
        data_set.append(data)
        print("Loaded feature {}".format(idx))
    return data_set


if __name__ == '__main__':
    
    np.random.seed(5)
    data_dir = "C:\\Users\\wsy\\Desktop\\dataset3"
    save_dir = "C:\\Users\\wsy\\Desktop\\dataset3"
    data_base = mfcc_loader(data_dir, frame_per_second=64,
                            mfcc_cof=20, mfcc_ord=14)
    # save_file(data_base, save_dir)
    save_data(save_dir, data_base, fname='mfcc.npy')
    # np.save(os.path.join(save_dir, 'data.npy'), np.vstack(data_base))
    print(len(data_base))
    print(len(data_base[5]))

