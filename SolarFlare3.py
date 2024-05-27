from dataSolarFlare3 import dataset
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score


def split_dataset_num_samples(dataset, num_samples):
    train_set = dataset[:num_samples]
    test_set = dataset[num_samples:]
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
    X = int(input())
    criteria = 'gini'
    classifier = DecisionTreeClassifier(criterion=criteria, random_state=0)

    num_samples = len(dataset) - X
    train_set, test_set = split_dataset_num_samples(dataset, num_samples)
    train_x, train_y = split_x_and_y(train_set)
    test_x, test_y = split_x_and_y(test_set)

    encoder = OrdinalEncoder()
    x, y = split_x_and_y(dataset)
    encoder.fit(x)

    classifier.fit(encoder.transform(train_x), train_y)
    classifier_predict = classifier.predict(encoder.transform(test_x))
    accuracy = accuracy_score(test_y, classifier_predict)
    print(f"Accuracy: {accuracy}")

    TP, FP, TN, FN = get_metrics(test_y, classifier_predict, "1", "0")
    denominator = TP + FP
    if denominator == 0:
        print(f"Precision: 0.0")
    else:
        precision = TP / denominator
        print(f"Precision: {precision}")

