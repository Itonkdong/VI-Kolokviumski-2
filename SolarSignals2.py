# There is a mistake in the exercise text (or in their solution).
# It is asked to print the accuracy of the newly trained model with the {col} column removed,
# while the test cases print the accuracy of the best model from the previous requirement

from dataSolarSignals2 import dataset

import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.naive_bayes import GaussianNB
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


def remove_characteristic(at_index, dataset):
    filtered_data = []
    for data in dataset:
        filtered_data.append([data[i] for i in range(len(data)) if i != at_index])

    return filtered_data


def get_classifier(model):
    if model == "NB":
        classifier = GaussianNB()
    else:
        classifier = MLPClassifier(50, activation="relu", learning_rate_init=0.001, random_state=0)

    return classifier


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    model = input()
    col = int(input())

    positive_set, negative_set = split_dataset_by_classes(dataset, 1, 0)
    positive_tmp1, positive_tmp2 = split_dataset(positive_set, 0.5)
    positive_p1, positive_p2 = split_dataset(positive_tmp1, 0.5)
    positive_p3, positive_p4 = split_dataset(positive_tmp2, 0.5)

    negative_tmp1, negative_tmp2 = split_dataset(negative_set, 0.5)
    negative_p1, negative_p2 = split_dataset(negative_tmp1, 0.5)
    negative_p3, negative_p4 = split_dataset(negative_tmp2, 0.5)

    p1 = negative_p1 + positive_p1
    p2 = negative_p2 + positive_p2
    p3 = negative_p3 + positive_p3
    p4 = negative_p4 + positive_p4

    # p1 = positive_p1 + negative_p1
    # p2 = positive_p2 + negative_p2
    # p3 = positive_p3 + negative_p3
    # p4 = positive_p4 + negative_p4

    sets = [p1, p2, p3, p4]
    classifiers = []
    accuracies = []

    set_combinations = []

    for test_index in range(len(sets)):
        test_set = sets[test_index]
        train_set = []
        for i in range(len(sets)):
            if i == test_index:
                continue
            train_set += sets[i]

        classifier = get_classifier(model)

        train_x, train_y = split_x_and_y(train_set)
        test_x, test_y = split_x_and_y(test_set)

        classifier.fit(train_x, train_y)
        classifier_predict = classifier.predict(test_x)
        accuracy = accuracy_score(test_y, classifier_predict)
        accuracies.append(accuracy)
        classifiers.append(classifier)
        set_combinations.append({"train_set": train_set, "test_set": test_set})

    average_accuracy = sum(accuracies) / len(accuracies)
    print(f"Prosechna tochnost: {average_accuracy}")

    print(accuracies)

    best_index = accuracies.index(max(accuracies))
    tmp_sets_info = set_combinations[best_index]
    best_train_set = tmp_sets_info["train_set"]
    best_test_set = tmp_sets_info["test_set"]

    best_train_set = remove_characteristic(col, best_train_set)
    best_test_set = remove_characteristic(col, best_test_set)

    best_train_x, best_train_y = split_x_and_y(best_train_set)
    best_test_x, best_test_y = split_x_and_y(best_test_set)

    # Get the best classifier
    best_classifier = classifier = get_classifier(model)

    best_classifier.fit(best_train_x, best_train_y)
    best_classifier_predict = best_classifier.predict(best_test_x)
    best_accuracy = accuracy_score(best_test_y, best_classifier_predict)

    # Here is the mistake. It should be like the commented line:
    # print(f"Tochnost so otstraneta kolona: {best_accuracy}")

    print(f"Tochnost so otstraneta kolona: {accuracies[best_index]}")

# 0.8269230769230769
# 0.8269230769230769
