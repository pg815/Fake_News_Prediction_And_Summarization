from getEmbeddings import getEmbeddings
from sklearn.naive_bayes import GaussianNB
import scikitplot.plotters as skplt
from sklearn.svm import SVC
import numpy as np
import pickle
import os

class Models:

    def __init__(self):
        if not os.path.isfile('./xtr.npy') or \
                not os.path.isfile('./xte.npy') or \
                not os.path.isfile('./ytr.npy') or \
                not os.path.isfile('./yte.npy'):
            xtr, xte, ytr, yte = getEmbeddings("datasets/train.csv")
            np.save('./xtr', xtr)
            np.save('./xte', xte)
            np.save('./ytr', ytr)
            np.save('./yte', yte)

        self.xtr = np.load('./xtr.npy')
        self.xte = np.load('./xte.npy')
        self.ytr = np.load('./ytr.npy')
        self.yte = np.load('./yte.npy')

    def train_svc_classifier(self):
        clf = SVC()
        clf.fit(self.xtr, self.ytr)
        #pickle.dump(clf, open("moddel.sav", "wb"))
        y_pred = clf.predict(self.xte)
        m = self.yte.shape[0]
        n = (self.yte != y_pred).sum()
        print("Accuracy of Support Vector Machine Classifier = " + format((m - n) / m * 100, '.2f') + "%")

    def train_nb_classifier(self):
        gnb = GaussianNB()
        gnb.fit(self.xtr, self.ytr)
        y_pred = gnb.predict(self.xte)
        m = self.yte.shape[0]
        n = (self.yte != y_pred).sum()
        print("Accuracy of Gaussian Naive Bayes Classifier = " + format((m - n) / m * 100, '.2f') + "%")  # 72.94%

    def predict_truthfullness(self,news):
        load_model = pickle.load(open('model.sav', 'rb'))
        prediction = load_model.predict([news])
        prob = load_model.predict_proba([news])

        return ["The given statement is " + str(prediction[0]),"The truth probability score is " + str(prob[0][1])]


if __name__ == "__main__":
    model = Models()
    # model.train_svc_classifier()
    # model.train_nb_classifier()
    result = model.predict_truthfullness("obama is running for president in 2022")
    print(result[0])
    print(result[1])

