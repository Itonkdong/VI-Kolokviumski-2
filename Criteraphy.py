from dataCriteraphy import dataset
from sklearn.naive_bayes import GaussianNB
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


def map_dataset(dataset, map_to_function=int, is_class_included=False):
    if not is_class_included:
        mapped = [list(map(map_to_function, data)) for data in dataset]
    else:
        mapped = []
        for data in dataset:
            partial_map = list(map(map_to_function, data[:-1]))
            partial_map.append(data[-1])
            mapped.append(partial_map)
    return mapped


if __name__ == '__main__':
    dataset = map_dataset(dataset, float, True)

    train_set, test_set = split_dataset(dataset, 0.85)
    train_x, train_y = split_x_and_y(train_set)
    test_x, test_y = split_x_and_y(test_set)

    classifier = GaussianNB()
    classifier.fit(train_x, train_y)
    classifier_predict = classifier.predict(test_x)
    accuracy = accuracy_score(test_y, classifier_predict)
    sample = list(map(float, input().split()))
    sample = [sample]
    sample_prediction = classifier.predict(sample)[0]
    probabilities = classifier.predict_proba(sample)
    print(accuracy)
    print(sample_prediction)
    print(probabilities)
