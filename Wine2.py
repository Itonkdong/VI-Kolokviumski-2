from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from dataWine2 import dataset


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


if __name__ == '__main__':
    x1 = float(input())
    x2 = float(input())
    class_1_set, class_0_set = split_dataset_by_classes(dataset, 1, 0)
    _, class_2_set = split_dataset_by_classes(dataset, 1, 2)

    bayes_classifier = GaussianNB()
    bayes_train0_set, _ = split_dataset(class_0_set, x1)
    bayes_train1_set, _ = split_dataset(class_1_set, x1)
    bayes_train2_set, _ = split_dataset(class_2_set, x1)
    bayes_train_set = bayes_train0_set + bayes_train1_set + bayes_train2_set
    # bayes_train_set = bayes_train2_set + bayes_train1_set + bayes_train0_set
    bayes_train_x, bayes_train_y = split_x_and_y(bayes_train_set)
    bayes_classifier.fit(bayes_train_x, bayes_train_y)

    # =========================================================

    tree_classifier = DecisionTreeClassifier(random_state=0)
    start_index = int(x1 * len(class_0_set))
    end_index = int(x2 * len(class_0_set))
    tree_train0_set = class_0_set[start_index:end_index]
    tree_train1_set = class_1_set[start_index:end_index]
    tree_train2_set = class_2_set[start_index:end_index]

    tree_train_set = tree_train0_set + tree_train1_set + tree_train2_set
    # tree_train_set = tree_train2_set + tree_train1_set + tree_train0_set
    tree_train_x, tree_train_y = split_x_and_y(tree_train_set)
    tree_classifier.fit(tree_train_x, tree_train_y)

    # =========================================================

    forest_classifier = RandomForestClassifier(n_estimators=3, random_state=0)
    forest_train0_set, _ = split_dataset(class_0_set, x2)
    forest_train1_set, _ = split_dataset(class_1_set, x2)
    forest_train2_set, _ = split_dataset(class_2_set, x2)
    forest_train_set = forest_train0_set + forest_train1_set + forest_train2_set
    # forest_train_set = forest_train2_set + forest_train1_set + forest_train0_set
    forest_train_x, forest_train_y = split_x_and_y(forest_train_set)
    forest_classifier.fit(forest_train_x, forest_train_y)

    # # =========================================================
    _, test0_set = split_dataset(class_0_set, x2)
    _, test1_set = split_dataset(class_1_set, x2)
    _, test2_set = split_dataset(class_2_set, x2)

    test_set = test0_set + test1_set + test2_set
    test_x, test_y = split_x_and_y(test_set)

    # # =========================================================

    bayes_classifier_predict = bayes_classifier.predict(test_x)
    tree_classifier_predict = tree_classifier.predict(test_x)
    forest_classifier_predict = forest_classifier.predict(test_x)

    total = 0
    for bayes_p, tree_p, forest_p, gt in zip(bayes_classifier_predict, tree_classifier_predict,
                                             forest_classifier_predict, test_y):
        hits = 0
        tmp_pred = [bayes_p, tree_p, forest_p]
        for prediction in tmp_pred:
            if prediction == gt:
                hits += 1
        if hits >= 2:
            total += 1

    accuracy = total / len(test_y)

    print(f"Tochnost: {accuracy}")
