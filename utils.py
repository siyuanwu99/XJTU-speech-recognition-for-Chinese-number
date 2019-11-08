import numpy as np
import librosa.display as lbdis
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from audio_processor import AudioProcessor
import os
from sklearn.metrics import confusion_matrix
from sklearn import manifold
import seaborn as sns


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


def t_sne(features, labels):
    '''
    t-distributed stochastic neighbor embedding for visualization data.
    :return:
    '''
    # tsne =
    X_embedding = manifold.TSNE(n_components=2,
                                init='pca',
                                random_state=501).fit_transform(features)
    print('Shape:', X_embedding.shape)
    scatter(X_embedding, labels)
    plt.show()


def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


def find_max_index(x, n: int=2):
    '''
    x: numpy array
    n: int
    index: numpy array
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
    feture 分析： 找提取特征中的最近邻，并比较
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

    # best_10 = np.zeros()
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


if __name__ == '__main__':
    save_path = "C:\\Users\\wsy\\Desktop\\dataset3\\new_mfcc.npy"
    if os.path.exists(os.path.join(save_path)):
        data_base = np.load(os.path.join(save_path))
        print("Using saved data base")
    else:
        print("Wrong Input")

    # np.random.shuffle(data_base)
    t_sne(data_base[:, :-1], data_base[:, -1])
    # print("Time domain")
    nearest_neighbour(data_base[:, :-1], data_base[:, -1], 20, 10)
    # print("Frequency domain")
    #
    # save_path = "C:\\Users\\wsy\\Desktop\\dataset3\\mfcc64.npy"
    # if os.path.exists(os.path.join(save_path)):
    #     data_base = np.load(os.path.join(save_path))
    #     print("Using saved data base")
    # else:
    #     print("Wrong Input")
    # nearest_neighbour(data_base[:, :-1], data_base[:, -1], 20, 10)
