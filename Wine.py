from dataWine import dataset
import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import warnings
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score


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


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=ConvergenceWarning)

    x = int(input())
    x = x / 100

    tmp_dataset = []
    for row in dataset:
        class_value = row.pop()
        y = None
        if class_value >= 5:
            y = 1.0
        else:
            y = 0.0
        row.append(y)
        tmp_dataset.append(row)

    dataset = tmp_dataset

    test_set, train_set = split_dataset(dataset, x)
    train_x, train_y = split_x_and_y(train_set)
    test_x, test_y = split_x_and_y(test_set)

    tree_classifier = DecisionTreeClassifier(criterion="gini", random_state=0)
    tree_classifier.fit(train_x, train_y)
    feature_importance = list(tree_classifier.feature_importances_)
    least_important_feature_index = feature_importance.index(min(feature_importance))

    nf_train_set = remove_characteristic(least_important_feature_index, train_set)
    nf_test_set = remove_characteristic(least_important_feature_index, test_set)

    nf_train_x, nf_train_y = split_x_and_y(nf_train_set)
    nf_test_x, nf_test_y = split_x_and_y(nf_test_set)

    mlp_classifier1 = MLPClassifier(15, activation="relu", learning_rate_init=0.001, max_iter=200, random_state=0)
    mlp_classifier2 = MLPClassifier(15, activation="relu", learning_rate_init=0.001, max_iter=200, random_state=0)

    standard_scaler = StandardScaler()
    min_max_scaler = MinMaxScaler()

    standard_scaler.fit(nf_train_x)
    min_max_scaler.fit(nf_train_x)

    mlp_classifier1.fit(standard_scaler.transform(nf_train_x), nf_train_y)
    mlp_classifier2.fit(min_max_scaler.transform(nf_train_x), nf_train_y)

    classifier1_predict = mlp_classifier1.predict(standard_scaler.transform(nf_test_x))
    classifier2_predict = mlp_classifier2.predict(min_max_scaler.transform(nf_test_x))

    accuracy1 = accuracy_score(nf_test_y, classifier1_predict)
    accuracy2 = accuracy_score(nf_test_y, classifier2_predict)

    print(f"Tocnost so StandardScaler: {accuracy1}")
    print(f"Tocnost so MinMaxScaler: {accuracy2}")

