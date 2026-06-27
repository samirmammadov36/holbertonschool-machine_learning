#!/usr/bin/env python3
"""Module that defines a random forest classifier."""

import numpy as np

Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Random_Forest:
    """Represent a random forest classifier."""

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """Initialize a random forest."""
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        """Predict the class of each individual."""
        predictions = np.array([
            tree_predict(explanatory)
            for tree_predict in self.numpy_preds
        ])

        classes = np.unique(self.target)

        class_counts = np.array([
            np.sum(predictions == class_value, axis=0)
            for class_value in classes
        ])

        return classes[np.argmax(class_counts, axis=0)]

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """Train the random forest."""
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []

        depths = []
        nodes = []
        leaves = []
        accuracies = []

        for i in range(n_trees):
            tree = Decision_Tree(
                max_depth=self.max_depth,
                min_pop=self.min_pop,
                seed=self.seed + i
            )

            tree.fit(explanatory, target)

            self.numpy_preds.append(tree.predict)
            depths.append(tree.depth())
            nodes.append(tree.count_nodes())
            leaves.append(tree.count_nodes(only_leaves=True))
            accuracies.append(
                tree.accuracy(tree.explanatory, tree.target)
            )

        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}
    - Mean accuracy on training data : {np.array(accuracies).mean()}
    - Accuracy of the forest on td   : {
        self.accuracy(self.explanatory, self.target)
    }""")

    def accuracy(self, test_explanatory, test_target):
        """Calculate the accuracy of the random forest."""
        predictions = self.predict(test_explanatory)

        return (
            np.sum(np.equal(predictions, test_target))
            / test_target.size
        )
