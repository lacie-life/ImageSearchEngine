import argparse
import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

class_names = []

train_image_list = []
train_id = []
train_img_bow = []
train_img_bow_label = []

test_image_list = []
test_id = []
test_img_bow = []
test_img_bow_label = []

descriptor_list = []

def labelLoader(path):
    global class_names
    class_names = os.listdir(path)


def readImage(img_path):
    # print(img_path)
    img = cv2.imread(img_path, 0)
    return cv2.resize(img, (150, 150))


def clusterDescriptors(descriptors, no_clusters, batch):
    #kmeans = KMeans(n_clusters=no_clusters).fit(descriptors)
    kmeans = MiniBatchKMeans(n_clusters=no_clusters, batch_size=batch, verbose=1).fit(descriptors)
    return kmeans


def getFiles(train, path):
    images = []
    for folder in os.listdir(path):
        for file in os.listdir(os.path.join(path, folder)):
            images.append(os.path.join(path, os.path.join(folder, file)))
    if (train is True):
        np.random.shuffle(images)
    return images


def plotConfusionMatrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def plotConfusions(true, predictions):
    np.set_printoptions(precision=2)

    plotConfusionMatrix(true, predictions, classes=class_names,
                      title='Confusion matrix, without normalization')

    plotConfusionMatrix(true, predictions, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

    plt.savefig("confusion_matrix.png")


def Run(train_path, test_path, no_clusters, batch):

    train_image_list = getFiles(True, train_path)
    test_image_list = getFiles(False, test_path)

    sift = cv2.xfeatures2d.SIFT_create()

    img_count = len(train_image_list)

    for img_path in train_image_list:

        print(img_path)

        img = readImage(img_path)

        kp, des = sift.detectAndCompute(img, None)

        for d in des:
            descriptor_list.append(d)

    kmeans = clusterDescriptors(descriptor_list, no_clusters, batch)

    # Create histogram of features for each training image
    for img_path in train_image_list:
        for i in range(0, len(class_names)):
            if class_names[i] in img_path:
                class_index = i

        img = readImage(img_path)

        kp, des = sift.detectAndCompute(img, None)

        histo = np.zeros(no_clusters)
        nkp = np.size(kp)  # nkp: number of keypoints

        for d in des:
            idx = kmeans.predict([d])
            histo[idx] += 1

        histo = np.array(histo)  # convert to vector numpy
        histo = histo / nkp  # normalized

        train_img_bow.append(histo)
        train_img_bow_label.append(class_index)


    # Create histogram of features for each testing image
    for img_path in test_image_list:
        print(img_path)
        for i in range(0, len(class_names)):
            if class_names[i] in img_path:
                class_index = i

        img = readImage(img_path)

        kp, des = sift.detectAndCompute(img, None)

        histo = np.zeros(no_clusters)
        nkp = np.size(kp)  # nkp: number of keypoints

        for d in des:
            idx = kmeans.predict([d])
            histo[idx] += 1

        histo = np.array(histo)  # convert to vector numpy
        histo = histo / nkp  # normalized

        test_img_bow.append(histo)
        test_img_bow_label.append(class_index)


    # Classification using SVM
    X = np.array(train_img_bow)
    Y = np.array(train_img_bow_label)

    X_test = np.array(test_img_bow)
    Y_test = np.array(test_img_bow_label)

    classifier = SVC(C=5, kernel='rbf', gamma='scale')
    classifier.fit(X, Y)
    res = classifier.predict(X_test)

    plotConfusions(test_img_bow_label, res)
    print("Confusion matrixes plotted.")

    accuracy = sum(res==Y_test)/len(Y_test)
    print(accuracy)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path', action="store", dest="train_path",
                        default="/home/jun/Github/BoVW/random/train")
    parser.add_argument('--test_path', action="store", dest="test_path",
                        default="/home/jun/Github/BoVW/random/test")
    parser.add_argument('--word', action="store", dest="word_num", default=500)
    parser.add_argument('--batch', action="store", dest="batch", default=3000)

    args = vars(parser.parse_args())

    labelLoader(args['train_path'])
    print(class_names)

    Run(args['train_path'], args['test_path'], int(args['word_num']), args['batch'])


