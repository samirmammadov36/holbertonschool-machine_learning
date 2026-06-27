def possible_thresholds(self, node, feature):
    """Return all possible thresholds for a feature."""
    values = np.unique(
        self.explanatory[:, feature][node.sub_population]
    )
    return (values[1:] + values[:-1]) / 2


def Gini_split_criterion_one_feature(self, node, feature):
    """Return the best threshold and Gini value for one feature."""
    thresholds = self.possible_thresholds(node, feature)

    if thresholds.size == 0:
        return 0, np.inf

    values = self.explanatory[:, feature][node.sub_population]
    targets = self.target[node.sub_population]
    classes = np.unique(targets)

    left_f = np.logical_and(
        values[:, None, None] > thresholds[None, :, None],
        targets[:, None, None] == classes[None, None, :]
    )

    right_f = np.logical_and(
        values[:, None, None] <= thresholds[None, :, None],
        targets[:, None, None] == classes[None, None, :]
    )

    left_counts = np.sum(left_f, axis=0)
    right_counts = np.sum(right_f, axis=0)

    left_sizes = np.sum(left_counts, axis=1)
    right_sizes = np.sum(right_counts, axis=1)
    population_size = targets.size

    left_gini = 1 - np.sum(
        (left_counts / left_sizes[:, None]) ** 2,
        axis=1
    )

    right_gini = 1 - np.sum(
        (right_counts / right_sizes[:, None]) ** 2,
        axis=1
    )

    gini_average = (
        left_sizes / population_size * left_gini
        + right_sizes / population_size * right_gini
    )

    best_index = np.argmin(gini_average)

    return thresholds[best_index], gini_average[best_index]


def Gini_split_criterion(self, node):
    """Return the feature and threshold with minimum Gini impurity."""
    results = np.array([
        self.Gini_split_criterion_one_feature(node, feature)
        for feature in range(self.explanatory.shape[1])
    ])

    feature = np.argmin(results[:, 1])

    return feature, results[feature, 0]
