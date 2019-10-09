import data_loader
import read_audio
import numpy as np
from sklearn import svm
import os
import time

NUM = 10


class AudioClassification(object):
    def __init__(self, classifier, data_dir, save_dir, num_clsfiers=20):

        self.N = NUM  # number of classes
        self.M = num_clsfiers  # number of classifiers
        self.classifier = classifier  # name of classifier
        if os.path.exists(os.path.join(save_dir, 'data.npy')):
            data_base = np.load(os.path.join(save_dir, 'data.npy'))
        else:
            data_base = data_loader.data_loader(data_dir,
                                                feature_length=10,
                                                frame_per_second=100)  # list of numpy
        data_base = np.vstack(data_base)
        np.random.shuffle(data_base)
        # num_validation = int(0.1 * len(data_base))
        self.num_train_set = int(0.9 * len(data_base))
        self.train_set = data_base[0:self.num_train_set]
        self.val_set = data_base[self.num_train_set:]
        self.code_book = self.get_class_codebook()
        self.clsfier_list = []

    def trainer(self):

        for idx in range(self.M):
            ti = time.time()
            # get code
            positive_label = np.where(self.code_book[:, idx] > 0)[0]
            negative_label = np.where(self.code_book[:, idx] < 0)[0]
            Y = self._find_code(self.train_set, positive_label, negative_label)

            # train classifier
            clsfier = self.train_a_classifier(self.train_set[:, :-1], Y, num=1)
            self.clsfier_list.append(clsfier)
            tt = time.time()
            print('-'*5 + 'Trained classifier {}'.format(idx) + '-'*5)
            print('Time elapsed {} seconds'.format(tt - ti))

            # validate
            val = self._find_code(self.val_set, positive_label, negative_label)
            self._validate(clsfier, self.val_set, val)

    def _find_code(self, input_data, positive, negative):
        Y = np.zeros([input_data.shape[0], 1])
        for i in range(input_data.shape[0]):
            cls = input_data[i, -1].astype(np.int64)
            if cls in positive:
                Y[i] += 1
            elif cls in negative:
                Y[i] -= 1
            else:
                print('ERROR in code book')
                assert 0, 'ERROR in code book'
        return Y

    def train_a_classifier(self, data, label, num=100):
        '''
        Should add other classifiers
        right now SVM only .
        :param data:
        :param label:
        :return:
        '''
        # if self.classifier == 'svm':
        #     clsfier = svm.LinearSVC()
        #     clsfier.fit(data, label)
        # elif self.classifier == 'decision_tree':
        #
        #     print("ERROR")
        clsfier = svm.SVC(kernel='rbf')
        for i in range(num):
            clsfier.fit(data, label)
        return clsfier

    def _validate(self, classifier, X, Y):
        predict = classifier.predict(X[:, :-1])
        error = np.abs(predict - Y.squeeze())
        accuracy = np.sum(error) / len(error) * 0.5

        print("Validation: accuracy = {}".format(accuracy))
        return

    def test(self):
        codebook = self.code_book.copy()
        predict_code = []
        for idx in range(self.M):
            cls = self.clsfier_list[idx]
            pdct = cls.predict(self.val_set[:, :-1])
            predict_code.append(pdct)

        predict_code = np.vstack(predict_code).transpose()  # N * M

        # give real code
        real_code = codebook.copy()
        cls = np.zeros(self.val_set.shape[0])
        for i, pdct in enumerate(predict_code):  # pdct: 1 * M
            dis = np.linalg.norm(real_code - pdct, axis=1)  #
            cls[i] = np.argmax(dis)
        print("Predict:\t\t", cls)
        print("Ground Truth:\t", self.val_set[:, -1])
        result = (cls - self.val_set[:, -1]).astype(np.int64)
        print(result)
        accuracy = 1 - np.count_nonzero(result) / len(result)
        print("ACCURACY:", accuracy)
        return accuracy

    def get_class_codebook(self):
        code_book = []
        for idx in range(self.M):
            code_line = np.ones(self.N).astype(np.int)
            y_label = np.arange(self.N)
            choice = np.random.choice(y_label, int(self.N // 2 - 1))
            code_line[choice] *= -1
            code_book.append(code_line)
        code_book_np = np.vstack(code_book).transpose()
        return code_book_np



if __name__ == '__main__':
    np.random.seed(5)
    data_dir = "C:\\Users\\wsy\\Desktop\\data_set_z71"
    save_dir = "C:\\Users\\wsy\\Desktop"
    AC = AudioClassification('svm', data_dir, save_dir)
    AC.trainer()
    AC.test()
