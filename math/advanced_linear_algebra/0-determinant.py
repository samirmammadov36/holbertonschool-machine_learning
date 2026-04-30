#!/usr/bin/env python3
"""Determinant of a matrix"""


def determinant(matrix):
    """Calculates the determinant of a matrix"""

    # Type check
    if not isinstance(matrix, list) or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Empty list check
    if matrix == []:
        raise TypeError("matrix must be a list of lists")

    # 0x0 matrix
    if matrix == [[]]:
        return 1

    # Square check
    n = len(matrix)
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # 1x1 matrix
    if n == 1:
        return matrix[0][0]

    # 2x2 matrix
    if n == 2:
        return (matrix[0][0] * matrix[1][1] -
                matrix[0][1] * matrix[1][0])

    # Recursive calculation
    det = 0
    for col in range(n):
        submatrix = [
            row[:col] + row[col + 1:]
            for row in matrix[1:]
        ]

        sign = (-1) ** col
        det += sign * matrix[0][col] * determinant(submatrix)

    return det
