import numpy as np
import librosa.display as lbdis
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from audio_processor import AudioProcessor
import os
import glob
from sklearn.metrics import confusion_matrix
from sklearn import manifold
import seaborn as sns
import librosa as lb


def save_data(save_dir, data, fname='data.npy'):
    np.save(os.path.join(save_dir, fname), np.vstack(data))
    print("Data has been saved to {:s}/{:s}".format(save_dir, fname))


def visualize_waves(path, frame_per_second):
    AP = AudioProcessor(feature_length=30,
                        frame_per_second=frame_per_second,
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
    lbdis.waveplot(audio, sr=frame_per_second)
    plt.title('windowed')
    plt.show()

    plt.figure(2)
    lbdis.waveplot(origin, sr=48000)
    plt.title("origin")
    plt.show()

    plt.figure(3)
    lbdis.waveplot(avg_zero_rate, sr=frame_per_second)
    plt.title('azr')
    plt.show()

    plt.figure(4)
    lbdis.waveplot(energy, sr=frame_per_second)
    plt.title('energy')
    plt.show()

    plt.figure(5)
    lbdis.waveplot(audio[boundary[0]:boundary[1] + 1], sr=frame_per_second)
    plt.title('cropped_avg')
    plt.show()

    plt.figure(6)
    lbdis.waveplot(energy[boundary[0]:boundary[1] + 1], sr=frame_per_second)
    plt.title('cropped energy')
    plt.show()

    print(len(audio))
    print("number of features", len(features))
    print(features[0])
    features = np.array(features)
    print(features.shape)


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(np.floor(cm * 10000)/100)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    return ax


def t_sne(features, labels, fname='t-sne.png'):
    '''
    t-distributed stochastic neighbor embedding for visualization data.
    :return:
    '''
    # tsne =
    X_embedding = manifold.TSNE(n_components=2,
                                init='pca',
                                learning_rate=600
                                ).fit_transform(features)
    print('Shape:', X_embedding.shape)
    scatter(X_embedding, labels)
    plt.savefig(fname)
    plt.show()


def scatter(x, colors):
    # We choose a color palette with seaborn.
    # palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    # ax = plt.subplot(aspect='equal')
    plt.scatter(x[:,0], x[:,1], c=colors, s=40, lw=0,
                cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar()
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)

    plt.axis('off')
    plt.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = plt.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)


def find_max_index(x, n: int=2):
    '''
    find max n index from the give 1-D matrix
    :param x: numpy array
    :param n: int
    :return index: numpy array
    '''
    x_ = x.copy()
    all_idx = np.arange(0, len(x)).tolist()
    index = []
    for _ in range(n):
        idx = np.argmax(x_)
        index.append(idx)
        x_[idx] = np.min(x)
        all_idx.remove(idx)
    return np.array(index).astype(np.int), np.array(all_idx).astype(np.int)


def nearest_neighbour(data_base, label, num_point: int=3, num_best: int=10):
    '''
    Nearest neighbour test to find the best feature
    feature 分析： 找提取特征中的最近邻，并比较
    :param data:
    :param label:
    :param num_point:
    :return:
    '''
    data_list = [[] for _ in range(10)]
    flag = np.zeros(10)
    s = 0

    while min(flag) < num_point:
        s += 1
        l = int(label[s]) # label

        if flag[l] < num_point:
            data_list[l].append(data_base[s])
            flag[l] += 1

    for i in range(10):
        sum = 0
        print('Label {:}'.format(i), end='\t')
        for data in data_list[i]:
            dis = -1 * np.sum(np.square(data_base - data), axis=1).squeeze()
            best_ten, _ = find_max_index(dis, num_best + 1)
            best_ten_label = label[best_ten[1:]].astype(np.int)
            a = num_best - np.count_nonzero(best_ten_label - i)
            sum += a
            # print('\t', best_ten_label, '\t', a)
        print('\tNearest Neighbour\t{}/{} \t {}'.
              format(sum, num_best*num_point, sum/num_best/num_point))


def pca_analysis(data, label):
    from sklearn.decomposition import PCA
    pca = PCA(4)
    pca.fit(data, label)
    new_data = pca.transform(data)
    new_data_base = np.hstack([new_data, label[:, np.newaxis]])
    return new_data_base


def waveplot_all(data_dir, NUM=10):
    """
    load all data from data_dir
    :param data_dir:
    :return: list of array
    """
    data_set = []
    for idx in range(NUM):
        data_path = os.path.join(data_dir, '*', '{}'.format(idx), '*.wav')
        file_list = glob.glob(data_path)

        # get audio data from file
        for file_path in file_list:
            wave, sr = lb.load(file_path, sr=None)
            lbdis.waveplot(wave, sr=sr)
            plt.title(file_path)
            plt.savefig(file_path.replace('wav', 'png'))
            plt.close()

            print("saved fig {}".format(file_path))

    return data_set


if __name__ == '__main__':
    # data_dir = "C:\\Users\\wsy\\Desktop\\dataset3"
    # waveplot_all(data_dir)
    save_path = "C:\\Users\\wsy\\Desktop\\dataset3\\mfcc128_25.npy"
    if os.path.exists(os.path.join(save_path)):
        data_base = np.load(os.path.join(save_path))
        print("Using saved data base")
    else:
        print("Wrong Input")

    np.random.shuffle(data_base)
    data_base = pca_analysis(data_base, data_base[:, -1])
    nearest_neighbour(data_base[:, :-1], data_base[:, -1], 20, 10)
    t_sne(data_base[:, :-1], data_base[:, -1], fname="C:\\Users\\wsy\\Desktop\\dataset3\\mfccts128.png")


