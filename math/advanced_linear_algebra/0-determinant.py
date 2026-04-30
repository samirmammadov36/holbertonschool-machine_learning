#!/usr/bin/env python3
"""Determinant of a matrix"""


def determinant(matrix):
    """Calculates the determinant of a matrix"""

    # Type check
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Empty matrix check
    if matrix == []:
        raise TypeError("matrix must be a list of lists")

    # 0x0 matrix
    if matrix == [[]]:
        return 1

    # Square check
    n = len(matrix)
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # 1x1
    if n == 1:
        return matrix[0][0]

    # 2x2 (shortcut)
    if n == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

    # Recursive Laplace expansion
    det = 0
    for col in range(n):
        # Submatrix (minor)
        sub = [
            row[:col] + row[col+1:]
            for row in matrix[1:]
        ]

        # Cofactor sign
        sign = (-1) ** col

        det += sign * matrix[0][col] * determinant(sub)

    return det
