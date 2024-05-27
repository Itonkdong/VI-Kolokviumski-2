from dataSolarFlare2 import dataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder


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
    percentage = int(input())
    criteria = input()
    percentage = 100 - percentage
    percentage = percentage / 100
    test_set, train_set = split_dataset(dataset, percentage)
    train_x, train_y = split_x_and_y(train_set)
    test_x, test_y = split_x_and_y(test_set)

    x, y = split_x_and_y(dataset)
    encoder = OrdinalEncoder()
    encoder.fit(x)
    classifier = DecisionTreeClassifier(criterion=criteria, random_state=0)
    classifier.fit(encoder.transform(train_x), train_y)
    classifier_predict = classifier.predict(encoder.transform(test_x))
    depth = classifier.get_depth()
    leaves = classifier.get_n_leaves()
    accuracy = accuracy_score(test_y, classifier_predict)
    feature_importance = list(classifier.feature_importances_)
    most_important_feature_index = feature_importance.index(max(feature_importance))
    least_important_feature_index = feature_importance.index(min(feature_importance))
    print(f"Depth: {depth}")
    print(f"Number of leaves: {leaves}")
    print(f"Accuracy: {accuracy}")
    print(f"Most important feature: {most_important_feature_index}")
    print(f"Least important feature: {least_important_feature_index}")
