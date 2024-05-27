from dataSolarFlare4 import dataset
from sklearn.neural_network import MLPClassifier


def split_dataset_num_samples(dataset, num_samples):
    train_set = dataset[:num_samples]
    test_set = dataset[num_samples:]
    return train_set, test_set


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
    x_num = int(input())

    classifier = MLPClassifier(3, activation='relu', learning_rate_init=0.003, max_iter=200, random_state=0)

    num_samples = len(dataset) - x_num
    train_set, test_set = split_dataset_num_samples(dataset, num_samples)
    train_x, train_y = split_x_and_y(train_set)
    test_x, test_y = split_x_and_y(test_set)

    classifier.fit(train_x, train_y)
    classifier_predict = classifier.predict(test_x)
    tp, fp, tn, fn = get_metrics(test_y, classifier_predict, 1, 0)

    precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
