# IMPORTANT: THE ORDER IN WHICH WE TAKE THE POSITIVE AND NEGATIVE SAMPLES IN THE GROUP TEST/TRAIN SET MATTERS!!!
# POSITIVE + NEGATIVE != NEGATIVE + POSITIVE, AT LEAST FOR THE FOREST CLASSIFIER

import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import warnings
from dataSolarSignals import dataset
from sklearn.exceptions import ConvergenceWarning
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


def split_dataset_by_classes(dataset, positive_class, negative_class, class_position=-1):
    positive_set = list(filter(lambda row: row[class_position] == positive_class, dataset))
    negative_set = list(filter(lambda row: row[class_position] == negative_class, dataset))
    return positive_set, negative_set


def split_dataset(dataset, percentage):
    train_upper_limit = int(percentage * len(dataset))
    train_set = dataset[:train_upper_limit]
    test_set = dataset[train_upper_limit:]

    return train_set, test_set


def split_x_and_y(dataset, class_position=-1):
    if class_position == -1:
        x = [row[:-1] for row in dataset]
        y = [row[-1] for row in dataset]
    else:
        x = [row[1:] for row in dataset]
        y = [row[0] for row in dataset]
    return x, y


def get_metrics(ground_truths, predictions, positive_class, negative_class):
    TP, TN, FP, FN = 0, 0, 0, 0

    for gt, prediction in zip(ground_truths, predictions):
        if gt == positive_class:
            if prediction == positive_class:
                TP += 1
            else:
                FN += 1
        else:
            if prediction == negative_class:
                TN += 1
            else:
                FP += 1

    return TP, FP, TN, FN


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    mode = input()
    split = int(input())
    split = split / 100
    if mode == "balanced":
        positive_set, negative_set = split_dataset_by_classes(dataset, 1, 0)
        positive_train_set, positive_test_set = split_dataset(positive_set, split)
        negative_train_set, negative_test_set = split_dataset(negative_set, split)
        train_set = negative_train_set + positive_train_set
        test_set = negative_test_set + positive_test_set

        # train_set = positive_train_set + negative_train_set
        # test_set = positive_test_set + negative_test_set
    else:
        train_set, test_set = split_dataset(dataset, split)

    train_x, train_y = split_x_and_y(train_set)
    test_x, test_y = split_x_and_y(test_set)

    bayes_classifier = GaussianNB()
    forest_classifier = RandomForestClassifier(n_estimators=50, criterion="entropy", random_state=0)
    mlp_classifier = MLPClassifier(hidden_layer_sizes=50, activation="relu", learning_rate_init=0.001, random_state=0)

    bayes_classifier.fit(train_x, train_y)
    forest_classifier.fit(train_x, train_y)
    mlp_classifier.fit(train_x, train_y)

    classifiers = [bayes_classifier, forest_classifier, mlp_classifier]
    precisions = []

    for classifier in classifiers:
        classifier_predict = classifier.predict(test_x)
        tp, fp, tn, fn = get_metrics(test_y, classifier_predict, 1, 0)
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        precisions.append(precision)

    most_precise_index = precisions.index(max(precisions))
    best_classifier = classifiers[most_precise_index]

    names = ["prviot", "vtoriot", "tretiot"]
    best_classifier_name = names[most_precise_index]
    best_classifier_predict = best_classifier.predict(test_x)

    accuracy = accuracy_score(test_y, best_classifier_predict)
    print(f"Najvisoka preciznost ima {best_classifier_name} klasifikator")
    print(f"Negovata tochnost e: {accuracy}")
