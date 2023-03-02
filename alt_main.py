import time

import numpy as np
from numpy import matrix as matrix
from numpy import array as vector
import copy


def flatten(multi_list):
    result = []
    for element in multi_list:
        if hasattr(element, "__iter__") and not isinstance(element, str):
            result.extend(flatten(element))
        else:
            result.append(element)
    return result


def _tau(mat: matrix) -> int:
    tau: int = 0
    for column in mat.T:
        if np.sum(column) > tau:
            tau = np.sum(column)
    return tau


# Diese Funktion ermittelt Spaltenmaximum der gegebenen Matrix in O(n)

def list_of_t_frequent_columns(mat: matrix, t: int) -> list:
    t_list: list = []
    m, n = mat.shape
    for i in range(n):
        if np.sum(mat.T[i]) == t:
            t_list.append(i)
    return t_list


def list_of_t_plus_frequent_columns(mat: matrix, t: int) -> list:
    t_list: list = []
    m, n = mat.shape
    for i in range(n):
        if np.sum(mat.T[i]) >= t:
            t_list.append(i)
    return t_list


# Diese Funktion gibt für gegebenes t eine Menge C an Spalten c  mit support R(c)= t zurück in O(n)

def closure(c, mat: matrix):
    m, n = mat.shape
    closed_set = []
    r: list = []
    v: vector = np.ones((m,), dtype=int)
    for i in c:
        v = np.multiply(v, np.asarray(mat.T[i]).reshape(m, ))

    for l in range(m):
        if v.flat[l] > 0:
            r.append(l)

    v: vector = np.ones(n, dtype=int)
    for k in r:
        v = np.multiply(v, np.asarray(mat[k]).reshape(n, ))
    for l in range(n):
        if v.flat[l] > 0:
            closed_set.append(l)

    return closed_set


# Diese Funktion ermittelt für gegebene Spaltenmenge die abgeschlossene Menge in O(|C|*m*n)


def all_subsets(nums, form="list"):
    subsets = [[]]

    for n in nums:
        prev = copy.deepcopy(subsets)
        [k.append(n) for k in subsets]
        subsets.extend(prev)
    del subsets[-1]

    # print("subsets",subsets)
    return subsets


# Diese Funktion ermittelt für gegebene Indexmenge alle Permutationen in O(n)

def support(c, mat):
    m, n = mat.shape
    v = np.ones((1, m), dtype=int)
    for i in c:
        v *= (mat[:, i].reshape(1, m))
    return np.sum(v)

def frequency(x, y):
    return 0


def boros_algorithm(mat: matrix, threshold: int):
    m, n = mat.shape
    tau: int = _tau(data)
    D = [[] for x in range(n)]
    count = 0
    for i in range(tau, threshold - 1, -1):
        t_list = list_of_t_frequent_columns(mat, i)
        t_plus_list = list_of_t_plus_frequent_columns(mat, i-1)
        subsets = all_subsets(t_list)
        for k in subsets:  # Gruppe (2)
            closed_set = closure(k, mat)
            support_c = support(closed_set, mat)
            if closed_set not in D[support_c - 1]:
                D[support_c - 1].append(closed_set)
                count += 1
    for i in range(tau-1, threshold - 1, -1):
        for closed_set in D[i]:

            t_plus_list = list_of_t_plus_frequent_columns(mat, i-1)
            x = list(set(np.arange(n)).difference(set(closed_set)))
            for column in x:
                new_closed_set = closure(flatten([column, closed_set]), mat)
                new_support_c = support(new_closed_set, mat)
                if new_closed_set not in D[new_support_c]:
                    D[new_support_c].append(new_closed_set)
                for k in range(1, support_c):
                    contenders = [closed_set]
                    if support(flatten([column, closed_set]), mat) >= support_c-k:
                        contenders.append(column)
                    subsets = all_subsets(contenders)[::2]
                    for subset in subsets:
                        new_closed_set = closure(flatten(subset), mat)
                        support_c = support(new_closed_set, mat)
                        if new_closed_set not in D[support_c - 1]:
                            D[support_c - 1].append(new_closed_set)
                            count += 1

    print(f"{count} closed sets bei {m}x{m}Matrix")

    return D


def test_setup(filename: str, matrix_data=np.array([])):
    global data
    if matrix_data.size == 0:
        data = np.loadtxt(filename, dtype=bool, delimiter=",")
    else:
        data = matrix_data


def algo_wrapper(threshold: int):
    boros_algorithm(data, threshold)



if __name__ == '__main__':
    u = 0
    np.random.seed(u)
    data: matrix = matrix(np.random.randint(0, 2, (15,15)))
    # print(data)
    output = "boros_output.txt"
    input = "test_data_2.txt"
    test_setup(filename="", matrix_data=data)
    algo_wrapper(0)


'''
8 [{1, 3, 4, 5, 6, 7, 9, 10, 11, 12}, {0, 1, 2, 4, 6, 8, 9, 11, 12}, {0, 3, 5, 6, 7, 8, 9, 10, 12}, {2, 4, 5, 6, 7, 8, 10, 12}, {0, 1, 2, 4, 7, 8, 10, 12}, {0, 1, 3, 6, 8, 9, 12}, {1, 2, 5, 10, 11}, {1, 2, 4, 5, 6, 7, 8, 9, 10}]
23 [{1, 4, 6, 9, 11, 12}, {3, 5, 6, 7, 9, 10, 12}, {5, 6, 7, 8, 10, 12}, {2, 4, 7, 8, 10, 12}, {0, 7, 8, 10, 12}, {4, 5, 6, 7, 10, 12}, {1, 4, 7, 10, 12}, {1, 3, 4, 5, 6, 10, 12}, {0, 3, 6, 8, 9, 12}, {0, 1, 6, 8, 9, 12}, {0, 3, 6, 7, 9, 12}, {1, 3, 6, 9, 12}, {2, 4, 6, 8, 12}, {0, 1, 2, 4, 8, 12}, {10, 11, 5, 6}, {1, 10, 11, 5}, {1, 2, 11}, {5, 6, 7, 8, 9, 10}, {1, 4, 5, 6, 7, 9, 10}, {2, 4, 5, 6, 7, 8, 10}, {1, 2, 4, 7, 8, 10}, {1, 2, 10, 5}, {1, 2, 4, 6, 8, 9}]
30 [{8, 10, 12, 7}, {5, 6, 7, 10, 12}, {10, 4, 12, 7}, {4, 5, 6, 10, 12}, {3, 5, 6, 10, 12}, {1, 10, 4, 12}, {3, 6, 7, 9, 12}, {0, 3, 6, 9, 12}, {1, 12, 6, 9}, {8, 2, 4, 12}, {0, 1, 12, 8}, {0, 12, 7}, {3, 4, 12, 6}, {1, 4, 12, 6}, {1, 3, 12, 6}, {10, 11, 5}, {11, 6}, {1, 11}, {5, 6, 7, 9, 10}, {5, 6, 7, 8, 10}, {2, 4, 7, 8, 10}, {4, 5, 6, 7, 10}, {1, 10, 4, 7}, {1, 4, 5, 6, 10}, {2, 10, 5}, {1, 2, 10}, {8, 1, 6, 9}, {1, 4, 6, 9}, {8, 2, 4, 6}, {8, 1, 2, 4}]
22 [{10, 12, 7}, {10, 12, 5, 6}, {10, 4, 12}, {0, 6, 8, 9, 12}, {9, 3, 12, 6}, {12, 6, 7}, {1, 12, 6}, {1, 4, 12}, {11}, {8, 10, 7}, {10, 5, 6, 7}, {10, 4, 7}, {10, 4, 5, 6}, {1, 10, 5}, {1, 10, 4}, {2, 10}, {9, 6, 7}, {1, 6, 9}, {8, 2, 4}, {8, 1}, {1, 4, 6}, {1, 2}]
15 [{10, 12}, {0, 9, 12, 6}, {8, 12, 6}, {0, 8, 12}, {12, 7}, {4, 12, 6}, {1, 12}, {10, 7}, {10, 4}, {1, 10}, {8, 9, 6}, {6, 7}, {1, 6}, {1, 4}, {2}]
9 [{9, 12, 6}, {8, 12}, {3, 12, 6}, {4, 12}, {0, 12}, {10, 5, 6}, {8, 6}, {7}, {4, 6}]
5 [{10, 5}, {9, 6}, {8}, {4}, {1}]
1 [{10}]
1 [{12, 6}]
1 [{12}]
1 [{6}]
11
'''