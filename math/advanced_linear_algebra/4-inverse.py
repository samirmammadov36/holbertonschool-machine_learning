#!/usr/bin/env python3
"""Inverse matrix"""


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
    """Calculates minor matrix"""

    n = len(matrix)

    if n == 1:
        return [[1]]

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


def cofactor(matrix):
    """Calculates cofactor matrix"""

    n = len(matrix)

    if n == 1:
        return [[1]]

    minor_matrix = minor(matrix)
    cofactor_matrix = []

    for i in range(n):
        row = []
        for j in range(n):
            sign = (-1) ** (i + j)
            row.append(sign * minor_matrix[i][j])
        cofactor_matrix.append(row)

    return cofactor_matrix


def adjugate(matrix):
    """Calculates adjugate matrix"""

    n = len(matrix)

    if n == 1:
        return [[1]]

    cof = cofactor(matrix)

    adj = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(cof[j][i])
        adj.append(row)

    return adj


def inverse(matrix):
    """Calculates the inverse of a matrix"""

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

    # Determinant
    det = determinant(matrix)

    if det == 0:
        return None

    # Adjugate
    adj = adjugate(matrix)

    # Divide by determinant
    inv = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(adj[i][j] / det)
        inv.append(row)

    return inv
