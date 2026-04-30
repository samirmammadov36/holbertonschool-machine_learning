#!/usr/bin/env python3
"""Minor matrix"""


def determinant(matrix):
    """Calculates determinant of a matrix"""

    if matrix == [[]]:
        return 1

    n = len(matrix)

    if n == 1:
        return matrix[0][0]

    if n == 2:
        return (matrix[0][0] * matrix[1][1] -
                matrix[0][1] * matrix[1][0])

    det = 0
    for col in range(n):
        submatrix = [
            row[:col] + row[col + 1:]
            for row in matrix[1:]
        ]
        sign = (-1) ** col
        det += sign * matrix[0][col] * determinant(submatrix)

    return det


def minor(matrix):
    """Calculates the minor matrix of a matrix"""

    # Type check
    if not isinstance(matrix, list) or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Empty or non-square
    if matrix == [] or matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)

    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # 1x1 case
    if n == 1:
        return [[1]]

    # Build minor matrix
    minor_matrix = []

    for i in range(n):
        row_minor = []
        for j in range(n):
            submatrix = [
                matrix[r][:j] + matrix[r][j + 1:]
                for r in range(n) if r != i
            ]
            row_minor.append(determinant(submatrix))
        minor_matrix.append(row_minor)

    return minor_matrix
