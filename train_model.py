import argparse
import glob
import utils
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report
from joblib import dump

def main(param):

    data_path = param.data_path
    numbers = range(1,7)
    names = ["ds", "ms", "mw"]

    X = []
    y = []

    for number in numbers:
        for name in names:
            for pathname in glob.glob(data_path + '/filmy_' + name + '_'  + str(number) + '_l.mov/*'):
                landmark = utils.read_landmark(os.path.join(pathname, "landmark.txt"))
                if (landmark[0] != 0.0):
                    X.append(landmark)
                    y.append(number -1)

    print(len(X), len(y))

    X, y = shuffle(X, y, random_state = 2)
    trainX, testX, trainY, testY = train_test_split(X, y, test_size = 0.3, random_state = 4)


    model = KNeighborsClassifier()
    #model = GaussianNB()
    model.fit(trainX, trainY)
    dump(model, 'model.joblib')

    y_pred = model.predict(testX)
    class_report = classification_report(testY, y_pred)
    conf_matrix = confusion_matrix(testY, y_pred)
    conf_matrix_printed = utils.print_confusion_matrix(conf_matrix, testY)

    # with open("report_naive_bayes_gaussian.txt", "w") as f:
    with open("report_5_neighbours.txt", "w") as f:
        f.write(class_report)
        f.write('\n')
        f.write(conf_matrix_printed)

    print(class_report)
    print(conf_matrix)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/npc/marta/mediapipe/npc_27_08_przyciete_227/",
        help="Directory of the images folder of our data set.",
    )



    main(param=parser.parse_args())
