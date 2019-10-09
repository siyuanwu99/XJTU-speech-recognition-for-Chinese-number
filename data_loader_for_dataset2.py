import numpy as np
import glob
import os
import json
from audio_processor import AudioProcessor

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
        # data_path = os.path.join(data_dir, '*', '{}.*'.format(idx))
        # for data dir person -> label
        data_path = os.path.join(data_dir, str(idx), '*')

        file_list = glob.glob(data_path)  # 所有文件名列表
        features_list = []

        # get audio data from file
        for file_path in file_list:
            A = AudioProcessor(frame_per_second, feature_length, file_path)
            features = A.get_cropped_feature()
            if features == [] or len(features) < 6 * feature_length:
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

    print('\n' + '-'*5 + 'NUM OF DATASET' + '-'*5 + '\n')
    for i, data_list in enumerate(data_set):
        print(str(i), '\t', len(data_list))
    return data_set


if __name__ == '__main__':
    
    np.random.seed(5)
    data_dir = "C:\\Users\\wsy\\Desktop\\data_set"
    save_dir = "C:\\Users\\wsy\\Desktop\\data_set"
    data_base = data_loader(data_dir,
                            feature_length=20,
                            frame_per_second=85)
    # save_file(data_base, save_dir)
    np.save(os.path.join(save_dir, 'data.npy'), np.vstack(data_base))
    print("Successfully generated data.npy")

