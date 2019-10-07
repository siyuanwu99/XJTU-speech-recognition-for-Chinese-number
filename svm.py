import numpy as np
from sklearn import svm
from data_loader import data_loader

N = 100
NUM_CLASS = 4

data_dir = "C:\\Users\\wsy\\Documents\\Audio\\*.m4a"
data_X, data_Y = data_loader(data_dir)
print(len(data_X))
clf_list = []


for idx in range(NUM_CLASS):
    for i, X in enumerate(data_X):
        if X == []:
            continue
        Y = data_Y[i][:, idx]
        clf = svm.LinearSVC()
        clf.fit(X, Y)

        # validation
        predicted = clf.predict(X[0:1, :])
        print("Predict:", predicted, "\t Ground Truth:", Y[0])
        clf_list.append(clf)

X = data_X[0]
test_X = X[3:4, :]
test_Y = data_Y[0][3:4, :]

test_predict = np.zeros(NUM_CLASS)
for idx in range(NUM_CLASS):
    clf = clf_list[idx]
    test_predict[idx] = clf.predict(test_X)

print("Predict:", test_predict, "\n Ground Truth:", test_Y)



