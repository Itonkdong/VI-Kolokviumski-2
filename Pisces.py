from dataPisces import dataset
from sklearn.ensemble import RandomForestClassifier
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
    col_index = int(input())
    trees = int(input())
    criteria = input()
    sample = list(map(float, input().split()))
    del sample[col_index]
    sample = [sample]

    dataset = remove_characteristic(col_index, dataset)
    train_set, test_set = split_dataset(dataset, 0.85)
    train_x, train_y = split_x_and_y(train_set)
    test_x, test_y = split_x_and_y(test_set)

    classifier = RandomForestClassifier(random_state=0, n_estimators=trees, criterion=criteria)
    classifier.fit(train_x, train_y)
    classifier_predict = classifier.predict(test_x)
    accuracy = accuracy_score(test_y, classifier_predict)
    sample_prediction = classifier.predict(sample)[0]
    probabilities = classifier.predict_proba(sample)[0]
    print(f"Accuracy: {accuracy}")
    print(sample_prediction)
    print(probabilities)
