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
        """Return a printable representation of the node and children."""
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

    def __str__(self):
        """Return a printable representation of the decision tree."""
        return self.root.__str__()
