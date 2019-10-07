import numpy as np
import glob
from read_audio import AudioProcessor


def generate_labels():
    """
    give labels for multiple classification
    :return:
    """
    label_1 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # >= 5
    label_2 = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # odd or even
    label_3 = [0, 0, 1, 1, 1, 1, 1, 0, 0, 0]  # >= 3 <= 7
    label_4 = [1, 1, 0, 0, 1, 1, 0, 0, 0, 1]  # randomly

    label = np.vstack([label_1, label_2, label_3, label_4]).transpose()

    return label


def load_data(path, frame_per_second=100):
    """
    load data from a file
    :param path:
    :param frame_per_second:
    :return:
    """
    AP = AudioProcessor(frame_per_second, path)
    # origin = AP.audio_data
    features = AP.get_feature()
    label = generate_labels()
    if not features == []:
        Transform_array = np.diag(np.ones([10]))
        np.random.shuffle(Transform_array)

        features_ = np.dot(Transform_array, features)
        label_ = np.dot(Transform_array, label)
    else:
        features_ = features
        label_ = []
    print("Loaded {}".format(path))

    return features_, label_

def data_loader(data_dir):
    file_list = glob.glob(data_dir)
    features_list = []
    label_list = []
    for file_path in file_list:
        features, labels = load_data(file_path)
        features_list.append(features)
        label_list.append(labels)
    return features_list, label_list


if __name__ == "__main__":
    np.random.seed(5)
    data_dir = "C:\\Users\\wsy\\Documents\\Audio\\*.m4a"
    features, labels = data_loader(data_dir)
    print(features)
    print(labels)
