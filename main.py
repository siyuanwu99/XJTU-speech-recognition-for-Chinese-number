import data_loader_for_dataset2 as data_loader
import audio_processor
from sklearn.multiclass import OutputCodeClassifier
import numpy as np
import sklearn
import os
import time
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

NUM = 10


class AudioClassification(object):
    def __init__(self, classifier, data_dir, save_path, num_clsfiers=20,
                 feature_length=20, frame_per_second=100, if_loaded=False):

        self.N = NUM  # 类的数目
        self.M = num_clsfiers  # 分类器的数目
        self.classifier = classifier  # 使用分类器的代号

        # 导入数据
        if os.path.exists(os.path.join(save_path)) and if_loaded:
            data_base = np.load(os.path.join(save_path))
            print("Using saved data base")
        else:
            data_base = data_loader.data_loader(data_dir,
                                                feature_length=feature_length,
                                                frame_per_second=frame_per_second)
        # dataloader: list of numpy
        data_base = np.vstack(data_base)
        np.random.shuffle(data_base)

        # 划分数据集
        # num_validation = int(0.1 * len(data_base))
        self.num_train_set = int(0.85 * len(data_base))
        num_test_set = int(0.95 * len(data_base))
        self.train_set = data_base[0:self.num_train_set]
        self.train_set_val = data_base[0: 50]  # 参与训练，就是看下分类器效果
        self.val_set = data_base[self.num_train_set:num_test_set]
        self.test_set = data_base[num_test_set:]

        self.code_book = self.get_class_codebook()  # 码本
        self.clsfier_list = []
        self.accuracy_list = []

    def trainer_ecoc(self):

        for idx in range(self.M):
            ti = time.time()
            # get code
            positive_label = np.where(self.code_book[:, idx] > 0)[0]
            negative_label = np.where(self.code_book[:, idx] < 0)[0]
            Y = self._find_code(self.train_set, positive_label, negative_label)

            # train classifier
            clsfier = self.train_a_classifier(self.train_set[:, :-1], Y, num=1)
            self.clsfier_list.append(clsfier)
            print('\n' + '-'*5 + ' Trained classifier {} '.format(idx) + '-'*5)

            print("for training set:")
            _ = self._validate(clsfier, self.train_set[0:40, :], Y[0:40])

            # validate
            print("for validation set:")
            val = self._find_code(self.val_set, positive_label, negative_label)
            accuracy = self._validate(clsfier, self.val_set, val)
            self.accuracy_list.append(accuracy)  # 将每个分类器的准确率存起来
            tt = time.time()
            print('\t\ttime elapsed {:.2f} seconds'.format(tt - ti))

    def trainer_multi_classifier(self):
        """
        Without ECOC code book
        :return:
        """

        ti = time.time()
        # train classifier
        self.clsfier = sklearn.tree.DecisionTreeClassifier()
        # self.clsfier = sklearn.svm.SVC(decision_function_shape='ovo')
        self.clsfier.fit(self.train_set[:, :-1], self.train_set[:, -1])
        self.clsfier_list.append(self.clsfier)
        print('\n' + '-'*5 + ' Trained classifier {} '.format(0) + '-'*5)

        print("for training set:")
        _ = self._validate(self.clsfier, self.train_set[0:40, :],
                           self.train_set[0:40, -1])

        # validate
        print("for validation set:")
        accuracy = self._validate(self.clsfier,
                                  self.val_set,
                                  self.val_set[:, -1])
        self.accuracy_list.append(accuracy)  # 将每个分类器的准确率存起来
        tt = time.time()
        print('\t\ttime elapsed {:.2f} seconds'.format(tt - ti))

        self.test_v2(self.val_set[:, :-1], self.val_set[:, -1])

    def reinforced_trainer(self, num_selected_classifiers):
        '''
        Using best num_selected_classifiers to regenerate the model
        :return:
        '''
        # select best classifiers
        accuracy = np.array(self.accuracy_list).copy()
        best_clf_list = []
        best_label = np.zeros(num_selected_classifiers)
        for i in range(num_selected_classifiers):
            idx = np.where(accuracy.max() == accuracy)[0][0]
            best_clf_list.append(self.clsfier_list[idx])
            best_label[i] = idx
            accuracy[idx] = 0
        best_label = best_label.astype(np.int64)
        accuracy = np.hstack(self.accuracy_list)[best_label]
        print('' * 20 + '\n')
        print('Select {}'.format(np.vstack([best_label, accuracy])))

        codebook = self.code_book[:, best_label.astype(np.int64)]
        print(codebook)
        predict_code = self._get_predict_code(best_clf_list)
        predict_code = np.vstack(predict_code).transpose()
        cls = np.zeros(self.test_set.shape[0])
        for i, pdct in enumerate(predict_code):  # pdct: 1 * M
            dis = np.sum(np.abs(codebook - pdct) * accuracy, axis=1)
            cls[i] = np.argmin(dis)  # 每个分类器的准确率有不同的权重

        self._print_val(cls)

    def _get_predict_code(self, clf_list: list) -> list:

        predict_code = []
        if len(clf_list) < 1:
            clf_list = self.clsfier_list
            print("No input, using default")
        for cls in clf_list:  # 经过每个分类器，然后预测对应的类别，组成ECOC码
            pdct = cls.predict(self.test_set[:, :-1])
            predict_code.append(pdct)
        return predict_code

    def _find_code(self, input_data, positive, negative):
        '''
        找这个label对应的ECOC码
        :param input_data:
        :param positive:
        :param negative:
        :return:
        '''
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

    def train_a_classifier(self, data, label, num=1000):
        '''
        Should add other classifiers
        right now SVM only .
        :param data:
        :param label:
        :return:
        '''
        if self.classifier == 'lsvm':
            clf = sklearn.svm.LinearSVC(max_iter=50000)
        elif self.classifier == 'ksvm':
            clf = sklearn.svm.SVC(kernel='sigmoid', gamma='scale', max_iter=50000)
            # kernel: ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
            # gamma: 'auto' 'scale'
        elif self.classifier == 'dctree':
            clf = sklearn.tree.DecisionTreeClassifier()
        elif self.classifier == 'sgd':
            clf = sklearn.linear_model.SGDClassifier(
                loss="modified_huber", penalty="l2")
        elif self.classifier == 'bayes':
            clf = sklearn.naive_bayes.GaussianNB()
        elif self.classifier == 'ada_boost':
            clf = AdaBoostClassifier(n_estimators=100)
        elif self.classifier == 'knn':
            clf = KNeighborsClassifier()
        else:
            clf = []
            assert 0, "ERROR"

        for i in range(num):
            clf.fit(data, label.squeeze())
        return clf

    def _validate(self, classifier, X, Y):
        """
        validate the classifier
        :param classifier:
        :param X:
        :param Y:
        :return:
        """
        predict = classifier.predict(X[:, :-1])
        error = np.abs(predict - Y.squeeze())
        accuracy = 1 - np.count_nonzero(error) / len(error)

        print("Validation: accuracy = {:.6f}".format(accuracy))
        return accuracy

    def test(self):

        predict_code = []
        for idx in range(self.M):  # 经过每个分类器，然后预测对应的类别，组成ECOC码
            cls = self.clsfier_list[idx]
            pdct = cls.predict(self.test_set[:, :-1])
            predict_code.append(pdct)

        predict_code = np.vstack(predict_code).transpose()  # N * M
        accuracy = np.hstack(self.accuracy_list)  # TODO 不确定度？

        # give real code
        real_code = self.code_book.copy()
        cls = np.zeros(self.test_set.shape[0])
        for i, pdct in enumerate(predict_code):  # pdct: 1 * M
            dis = np.sum(np.abs(real_code - pdct) * accuracy, axis=1)
            cls[i] = np.argmin(dis)  # 每个分类器的准确率有不同的权重

        self._print_val(cls)

    def _print_val(self, cls):
        print('\n' + '-' * 20 + '\nMethod: \t', self.classifier)
        print("Predict:\n", cls)
        print("Ground Truth:\n", self.test_set[:, -1])
        result = (cls - self.test_set[:, -1]).astype(np.int64)
        print("Error\n", result)
        accuracy = 1 - np.count_nonzero(result) / len(result)
        print("ACCURACY: {:.6f}".format(accuracy))
        return accuracy

    def test_v2(self, data, label):

        predict_code = self.clsfier.predict(data) # N * M
        # give real code
        print('\n' + '-' * 20 + '\nMethod: \t', self.classifier)
        print("Predict:\n", predict_code)
        print("Ground Truth:\n", label)
        result = (predict_code - label).astype(np.int64)
        print("Error\n", result)
        accuracy = 1 - np.count_nonzero(result) / len(result)
        print("ACCURACY: {:.6f}".format(accuracy))
        return accuracy

    def get_class_codebook(self):
        code_book = []
        for idx in range(self.M):
            code_line = np.ones(self.N).astype(np.int)
            y_label = np.arange(self.N)
            choice = np.random.choice(y_label, int(self.N // 2), replace=False)
            code_line[choice] *= -1
            code_book.append(code_line)
        code_book_np = np.vstack(code_book).transpose()
        # code_book_np = np.array([
        #     [1,1,0,0,0,0,1,0,1,0,0,1,1,0,1],
        #     [0,0,1,1,1,1,0,1,0,1,1,0,0,1,0],
        #     [1,0,0,1,0,0,0,1,1,1,1,0,1,0,1],
        #     [0,0,1,1,0,1,1,1,0,0,0,0,1,0,1],
        #     [1,1,1,0,1,0,1,1,0,0,1,0,0,0,1],
        #     [0,1,0,0,1,1,0,1,1,1,0,0,0,0,1],
        #     [1,0,1,1,1,0,0,0,0,1,0,1,0,0,1],
        #     [0,0,0,1,1,1,1,0,1,0,1,1,0,0,1],
        #     [1,1,0,1,0,1,1,0,0,1,0,0,0,1,1],
        #     [0,1,1,1,0,0,0,0,1,0,1,0,0,1,1]
        # ])
        # code_book_np = code_book_np * 2 - 1
        print('\n' + '-' * 5 + 'CODEBOOK' + '-' * 5 + '\yn')
        print(code_book_np.transpose())
        return code_book_np


if __name__ == '__main__':
    np.random.seed(4)
    data_dir = "C:\\Users\\wsy\\Desktop\\data_set"
    save_dir = "C:\\Users\\wsy\\Desktop\\data_set\\0_85.npy"
    AC = AudioClassification('dctree', data_dir, save_dir,
                             num_clsfiers=100,
                             feature_length=0,
                             frame_per_second=81,
                             if_loaded=False)
    # AC.trainer_multi_classifier()
    AC.trainer_ecoc()
    # AC.test()
    AC.reinforced_trainer(19)