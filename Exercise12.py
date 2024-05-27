from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from dataExcercise12 import dataset
import warnings


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

def my_accuracy_score(gts, predictions):
    total = 0
    for gt, prediction in zip(gts, predictions):
        if gt == prediction:
            total += 1

    return total / len(gts)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    neurons = int(input())
    learning_rate = float(input())
    col_index = int(input())
    sample = list(map(float, input().split()))

    classifier = MLPClassifier(neurons, learning_rate_init=learning_rate, activation="relu", max_iter=20,
                               random_state=0)

    scaler = MinMaxScaler((-1, 1))

    train_set, test_set = split_dataset(dataset, 0.8)
    train_x, train_y = split_x_and_y(train_set)
    test_x, test_y = split_x_and_y(test_set)

    scaler.fit(train_x)

    classifier.fit(scaler.transform(train_x), train_y)
    test_predict = classifier.predict(scaler.transform(test_x))
    train_predict = classifier.predict(scaler.transform(train_x))
    test_accuracy = my_accuracy_score(test_y, test_predict)
    train_accuracy = my_accuracy_score(train_y, train_predict)

    i = 0.15 * test_accuracy

    if train_accuracy - test_accuracy > i:
        print("Se sluchuva overfitting")
        classifier1 = MLPClassifier(neurons, learning_rate_init=learning_rate, activation="relu", max_iter=20,
                                   random_state=0)

        nf_train_set = remove_characteristic(col_index, train_set)
        nf_test_set = remove_characteristic(col_index, test_set)

        nf_train_x, nf_train_y = split_x_and_y(nf_train_set)
        nf_test_x, nf_test_y = split_x_and_y(nf_test_set)

        scaler1 = MinMaxScaler((-1, 1))
        scaler1.fit(nf_train_x)
        classifier1.fit(scaler1.transform(nf_train_x), nf_train_y)
        del sample[col_index]
        sample = [sample]
        sample_predict = classifier1.predict(scaler1.transform(sample))[0]
        print(sample_predict)
    else:
        print("Ne se sluchuva overfitting")
        sample = [sample]
        sample_predict = classifier.predict(scaler.transform(sample))[0]
        print(sample_predict)



