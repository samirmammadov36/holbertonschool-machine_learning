#!/usr/bin/env python3
"""Module containing classes for building a decision tree."""

import numpy as np


class Node:
    """Represent an internal node in a decision tree."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initialize a decision tree node."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth
        self.lower = None
        self.upper = None
        self.indicator = None

    def max_depth_below(self):
        """Return the maximum depth below the current node."""
        return max(
            self.left_child.max_depth_below(),
            self.right_child.max_depth_below()
        )

    def count_nodes_below(self, only_leaves=False):
        """Count nodes below the current node."""
        left_count = self.left_child.count_nodes_below(
            only_leaves=only_leaves
        )
        right_count = self.right_child.count_nodes_below(
            only_leaves=only_leaves
        )

        if only_leaves:
            return left_count + right_count

        return 1 + left_count + right_count

    def get_leaves_below(self):
        """Return all leaves below the current node."""
        return (
            self.left_child.get_leaves_below()
            + self.right_child.get_leaves_below()
        )

    def update_bounds_below(self):
        """Recursively update feature bounds below this node."""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -np.inf}

        self.left_child.lower = self.lower.copy()
        self.left_child.upper = self.upper.copy()

        if self.feature in self.left_child.lower:
            self.left_child.lower[self.feature] = max(
                self.left_child.lower[self.feature],
                self.threshold
            )
        else:
            self.left_child.lower[self.feature] = self.threshold

        self.right_child.lower = self.lower.copy()
        self.right_child.upper = self.upper.copy()

        if self.feature in self.right_child.upper:
            self.right_child.upper[self.feature] = min(
                self.right_child.upper[self.feature],
                self.threshold
            )
        else:
            self.right_child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """Create an indicator function using the node bounds."""
        def is_large_enough(x):
            """Check whether values exceed all lower bounds."""
            comparisons = np.array([
                np.greater(x[:, key], self.lower[key])
                for key in self.lower
            ])
            return np.all(comparisons, axis=0)

        def is_small_enough(x):
            """Check whether values satisfy all upper bounds."""
            comparisons = np.array([
                np.less_equal(x[:, key], self.upper[key])
                for key in self.upper
            ])
            return np.all(comparisons, axis=0)

        self.indicator = lambda x: np.all(
            np.array([
                is_large_enough(x),
                is_small_enough(x)
            ]),
            axis=0
        )

    def pred(self, x):
        """Predict the class of one individual."""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)

        return self.right_child.pred(x)

    def left_child_add_prefix(self, text):
        """Add the printing prefix used for a left child."""
        lines = text.rstrip("\n").split("\n")
        new_text = "    +--" + lines[0] + "\n"

        for line in lines[1:]:
            new_text += "    |  " + line + "\n"

        return new_text

    def right_child_add_prefix(self, text):
        """Add the printing prefix used for a right child."""
        lines = text.rstrip("\n").split("\n")
        new_text = "    +--" + lines[0] + "\n"

        for line in lines[1:]:
            new_text += "       " + line + "\n"

        return new_text

    def __str__(self):
        """Return a printable representation of the node."""
        if self.is_root:
            node_type = "root"
        else:
            node_type = "-> node"

        text = (
            f"{node_type} [feature={self.feature}, "
            f"threshold={self.threshold}]\n"
        )
        text += self.left_child_add_prefix(str(self.left_child))
        text += self.right_child_add_prefix(str(self.right_child))

        return text


class Leaf(Node):
    """Represent a leaf in a decision tree."""

    def __init__(self, value, depth=None):
        """Initialize a decision tree leaf."""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Return the depth of the leaf."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Count the current leaf."""
        return 1

    def get_leaves_below(self):
        """Return the current leaf in a list."""
        return [self]

    def update_bounds_below(self):
        """Stop recursive bounds updating at the leaf."""
        pass

    def pred(self, x):
        """Return the prediction stored in the leaf."""
        return self.value

    def __str__(self):
        """Return a printable representation of the leaf."""
        return f"-> leaf [value={self.value}]"


class Decision_Tree:
    """Represent a decision tree."""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initialize a decision tree."""
        self.rng = np.random.default_rng(seed)

        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)

        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Return the maximum depth of the decision tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Count all nodes or only leaves in the decision tree."""
        return self.root.count_nodes_below(
            only_leaves=only_leaves
        )

    def get_leaves(self):
        """Return all leaves in the decision tree."""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Compute bounds for all nodes in the tree."""
        self.root.update_bounds_below()

    def update_predict(self):
        """Create the vectorized prediction function."""
        self.update_bounds()
        leaves = self.get_leaves()

        for leaf in leaves:
            leaf.update_indicator()

        self.predict = lambda a: np.sum(
            np.array([
                leaf.indicator(a) * leaf.value
                for leaf in leaves
            ]),
            axis=0
        )

    def pred(self, x):
        """Predict the class of one individual."""
        return self.root.pred(x)

    def fit(self, explanatory, target, verbose=0):
        """Train the decision tree."""
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion

        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(
            self.target,
            dtype="bool"
        )

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {
        self.count_nodes(only_leaves=True)
    }
    - Accuracy on training data : {
        self.accuracy(self.explanatory, self.target)
    }""")

    def np_extrema(self, arr):
        """Return the minimum and maximum values of an array."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Select a random feature and threshold for a node."""
        diff = 0

        while diff == 0:
            feature = self.rng.integers(
                0,
                self.explanatory.shape[1]
            )
            values = self.explanatory[:, feature][
                node.sub_population
            ]
            feature_min, feature_max = self.np_extrema(values)
            diff = feature_max - feature_min

        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max

        return feature, threshold

    def fit_node(self, node):
        """Recursively train a node."""
        node.feature, node.threshold = self.split_criterion(node)
        values = self.explanatory[:, node.feature]

        left_population = np.logical_and(
            node.sub_population,
            values > node.threshold
        )
        right_population = np.logical_and(
            node.sub_population,
            values <= node.threshold
        )

        is_left_leaf = (
            np.sum(left_population) < self.min_pop
            or node.depth + 1 == self.max_depth
            or np.unique(self.target[left_population]).size == 1
        )

        if is_left_leaf:
            node.left_child = self.get_leaf_child(
                node,
                left_population
            )
        else:
            node.left_child = self.get_node_child(
                node,
                left_population
            )
            self.fit_node(node.left_child)

        is_right_leaf = (
            np.sum(right_population) < self.min_pop
            or node.depth + 1 == self.max_depth
            or np.unique(self.target[right_population]).size == 1
        )

        if is_right_leaf:
            node.right_child = self.get_leaf_child(
                node,
                right_population
            )
        else:
            node.right_child = self.get_node_child(
                node,
                right_population
            )
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """Create a leaf child for a sub-population."""
        classes, counts = np.unique(
            self.target[sub_population],
            return_counts=True
        )
        value = classes[np.argmax(counts)]

        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population

        return leaf_child

    def get_node_child(self, node, sub_population):
        """Create an internal node child for a sub-population."""
        child = Node()
        child.depth = node.depth + 1
        child.sub_population = sub_population

        return child

    def accuracy(self, test_explanatory, test_target):
        """Calculate prediction accuracy on a dataset."""
        return np.sum(
            np.equal(
                self.predict(test_explanatory),
                test_target
            )
        ) / test_target.size

    def possible_thresholds(self, node, feature):
        """Return all possible thresholds for a feature."""
        values = np.unique(
            self.explanatory[:, feature][node.sub_population]
        )
        return (values[1:] + values[:-1]) / 2

    def Gini_split_criterion_one_feature(self, node, feature):
        """Return the best threshold and Gini for one feature."""
        thresholds = self.possible_thresholds(node, feature)

        if thresholds.size == 0:
            return 0, np.inf

        values = self.explanatory[:, feature][
            node.sub_population
        ]
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

        return (
            thresholds[best_index],
            gini_average[best_index]
        )

    def Gini_split_criterion(self, node):
        """Return the feature and threshold with minimum Gini."""
        results = np.array([
            self.Gini_split_criterion_one_feature(
                node,
                feature
            )
            for feature in range(self.explanatory.shape[1])
        ])

        feature = np.argmin(results[:, 1])

        return feature, results[feature, 0]

    def __str__(self):
        """Return a printable representation of the tree."""
        return self.root.__str__()
