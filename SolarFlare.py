from dataSolarFlare import dataset
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
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


if __name__ == '__main__':
    x,y = split_x_and_y(dataset)
    encoder = OrdinalEncoder()
    encoder.fit(x)

    train_set,test_set = split_dataset(dataset, 0.75)
    train_x, train_y = split_x_and_y(train_set)
    test_x, test_y = split_x_and_y(test_set)

    classifier = CategoricalNB()
    classifier.fit(encoder.transform(train_x), train_y)
    classifier_predict = classifier.predict(encoder.transform(test_x))
    accuracy = accuracy_score(test_y, classifier_predict)
    sample = input().split()
    sample = encoder.transform([sample])
    sample_prediction = classifier.predict(sample)[0]
    probabilities = classifier.predict_proba(sample)
    print(accuracy)
    print(sample_prediction)
    print(probabilities)



