#!/usr/bin/env python3
"""Module containing classes for building a decision tree."""

import numpy as np


class Node:
    """Represent an internal node in a decision tree."""

    def __init__(
        self,
        feature=None,
        threshold=None,
        left_child=None,
        right_child=None,
        is_root=False,
        depth=0
    ):
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
        """Recursively update feature bounds below the current node."""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -np.inf}

        self.left_child.lower = self.lower.copy()
        self.left_child.upper = self.upper.copy()
        self.left_child.lower[self.feature] = self.threshold

        self.right_child.lower = self.lower.copy()
        self.right_child.upper = self.upper.copy()
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

    def __init__(
        self,
        max_depth=10,
        min_pop=1,
        seed=0,
        split_criterion="random",
        root=None
    ):
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

    def __str__(self):
        """Return a printable representation of the tree."""
        return self.root.__str__()
