# I GET DIFFERENT RESULTS LOCALLY, JUST A FRIENDLY REMINDER

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from dataSolarFlare5 import dataset


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


def remove_characteristic(at_index, dataset):
    filtered_data = []
    for data in dataset:
        filtered_data.append([data[i] for i in range(len(data)) if i != at_index])

    return filtered_data


def copy_dataset(dataset):
    copy = [row[:] for row in dataset]
    return copy


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


def get_classifier(model):
    if model == "NB":
        return GaussianNB()

    if model == "DT":
        return DecisionTreeClassifier(random_state=0)

    return MLPClassifier(3, activation="relu", learning_rate_init=0.003, max_iter=200, random_state=0)


def my_accuracy_score(gts, predictions):
    total = 0
    for gt, prediction in zip(gts, predictions):
        if gt == prediction:
            total += 1

    return total / len(gts)


if __name__ == '__main__':
    x = int(input())
    model = input()
    col = int(input())

    nf_dataset = remove_characteristic(col, copy_dataset(dataset))

    test_set, train_set = split_dataset_num_samples(dataset, x)
    # print(len(dataset))
    # print(len(test_set))
    # print(len(train_set))

    nf_test_set, nf_train_set = split_dataset_num_samples(nf_dataset, x)

    train_x, train_y = split_x_and_y(train_set)
    test_x, test_y = split_x_and_y(test_set)

    nf_train_x, nf_train_y = split_x_and_y(nf_train_set)
    nf_test_x, nf_test_y = split_x_and_y(nf_test_set)

    classifier1 = get_classifier(model)
    classifier2 = get_classifier(model)

    classifier1.fit(train_x, train_y)
    classifier1_predict = classifier1.predict(test_x)

    classifier2.fit(nf_train_x, nf_train_y)
    classifier2_predict = classifier2.predict(nf_test_x)

    accuracy1 = my_accuracy_score(test_y, classifier1_predict)
    accuracy2 = my_accuracy_score(nf_test_y, classifier2_predict)

    if accuracy1 == accuracy2:
        TP, FP, TN, FN = get_metrics(test_y, classifier1_predict, 1, 0)
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0.0
        print("Klasifikatorite imaat ista tochnost")
    elif accuracy1 > accuracy2:
        TP, FP, TN, FN = get_metrics(test_y, classifier1_predict, 1, 0)
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0.0
        print("Klasifiktorot so site koloni ima pogolema tochnost")
    else:
        TP, FP, TN, FN = get_metrics(nf_test_y, classifier2_predict, 1, 0)
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0.0
        print("Klasifiktorot so edna kolona pomalku ima pogolema tochnost")

    print(precision)
