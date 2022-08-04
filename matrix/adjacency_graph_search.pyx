#! python
# cython: language_level=3
import os
import time
from openpyxl import load_workbook
from matrix.adjacency_matrix import gen_nodes_from_scratch, create_adjacency_matrix_no_symbols,\
    load_nodes_and_matrix
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from matrix.nodes_to_snake import nodes_to_snake


nodes = []
matrix = []
prev_lengths = []

longuest_snakes = []
longuest_snake_length = 0

b, h = 0, 0


def init_nodes_and_matrix(load: bool):
    global nodes, matrix
    step_time = time.time()
    if load:
        nodes, matrix = load_nodes_and_matrix(b, False)
        if nodes is not None and matrix is not None:
            return
    print("Generating nodes...", end=" ")
    step_time = time.time()
    nodes = gen_nodes_from_scratch(b)
    print(f"Done in {(time.time() - step_time):.3f}s")
    print("Creating adjacency matrix...", end=" ")
    step_time = time.time()
    matrix = create_adjacency_matrix_no_symbols(nodes)
    print(f"Done in {(time.time() - step_time):.3f}s")
    step_time = time.time()


def print_longuest_snake():
    print("\nLength", longuest_snake_length)
    for node in longuest_snakes[0]:
        print(node)


def reset_snakes():
    global longuest_snakes, longuest_snake_length
    snake_length, snake_number = 0, 0
    snake = []
    longuest_snakes = []
    longuest_snake_length = 0


cdef init_csearch(int expected_number_of_snakes, forest):
    cdef int forest_c = 0, lenght_only = 0
    cdef int number_of_nodes = len(nodes) + 1
    cdef int i = 0, j = 0, n = 0
    cdef int longuest_snake_length_c = longuest_snake_length
    cdef int *snake_c = <int *> PyMem_Malloc(h * sizeof(int))
    cdef int *lengths_c = <int *> PyMem_Malloc(len(prev_lengths) * sizeof(int))
    cdef int *matrix_c = <int *> PyMem_Malloc(number_of_nodes * number_of_nodes * 2 * sizeof(int))
    cdef int *longuest_snakes_c = <int*> PyMem_Malloc(expected_number_of_snakes * h * sizeof(int))
    try:
        if forest:
            forest_c = 1
        if expected_number_of_snakes == 1:
            lenght_only = 1
        # Fills lengths_c
        for n in range(len(prev_lengths)):
            lengths_c[n] = prev_lengths[n]
        # Fills matrix_c
        for i in range(number_of_nodes):
            for j in range(number_of_nodes):
                if matrix[i][j] == 0:
                    matrix_c[i + number_of_nodes * (j + number_of_nodes * 0)] = -1
                    matrix_c[i + number_of_nodes * (j + number_of_nodes * 1)] = -1
                else:
                    matrix_c[i + number_of_nodes * (j + number_of_nodes * 0)] = matrix[i][j][0]
                    matrix_c[i + number_of_nodes * (j + number_of_nodes * 1)] = matrix[i][j][1]
        number_of_longuest_snakes = search_node(h, 0, 0, number_of_nodes, matrix_c, lengths_c, longuest_snakes_c,
                                                &longuest_snake_length_c, 0, snake_c, 0, 0, forest=forest_c, length_only=lenght_only)
        global longuest_snakes
        longuest_snakes = []
        snake = []
        for index in range(number_of_longuest_snakes * h):
            snake.append(nodes[longuest_snakes_c[index] - 1])
            if index != 0 and index % h == h - 1 or number_of_longuest_snakes * h == 1:
                longuest_snakes.append(snake)
                snake = []
    finally:
        PyMem_Free(lengths_c)
        PyMem_Free(longuest_snakes_c)
        PyMem_Free(matrix_c)
        PyMem_Free(snake_c)


cdef int search_node(int h, int current_h, int i, int number_of_nodes, int matrix[], int lengths[],
                     int longuest_snakes[], int* longuest_snake_length, int number_of_longuest_snakes,
                     int snake[], int snake_length, int snake_number, int forest, int length_only):
    cdef int n = 0, j = 0, h_left = 0, y = 0, z = 0
    for j in range(number_of_nodes - 1, -1, -1):
        y = matrix[i + number_of_nodes * (j + number_of_nodes * 0)]
        z = matrix[i + number_of_nodes * (j + number_of_nodes * 1)]
        if y != -1:
            snake[current_h] = j
            current_h += 1
            snake_length += y
            snake_number += z
            if current_h == h:
                if longuest_snake_length[0] <= snake_length and (forest == 1 or snake_number == 1):
                    n = 0
                    if longuest_snake_length[0] == snake_length and length_only == 0:
                        for n in range(h):
                            longuest_snakes[number_of_longuest_snakes * h + n] = snake[n]
                        number_of_longuest_snakes += 1
                    else:
                        longuest_snake_length[0] = snake_length
                        for n in range(h):
                            longuest_snakes[n] = snake[n]
                        number_of_longuest_snakes = 1
            elif current_h < h:
                h_left = h - current_h
                if snake_length + lengths[h_left - 1] >= longuest_snake_length[0]:
                    number_of_longuest_snakes = \
                        search_node(h, current_h, j, number_of_nodes, matrix, lengths, longuest_snakes,
                                    longuest_snake_length, number_of_longuest_snakes, snake, snake_length, snake_number,
                                    forest=forest, length_only=length_only)
            current_h -= 1
            snake_length -= y
            snake_number -= z
    return number_of_longuest_snakes


def graph_search(rect_dims: tuple, expected_number_of_snakes: int,
                 forest=False, sheet=None, load=True, verbose=True, max_length=0):
    global b, h, nodes, matrix, longuest_snakes, longuest_snake_length
    start_time = time.time()
    b, h = rect_dims
    longuest_snake_length = max_length
    init_prev_lengths(sheet)
    init_nodes_and_matrix(load)
    step_time = time.time()
    init_csearch(expected_number_of_snakes, forest)
    longuest_snake_length = nodes_to_snake(longuest_snakes[0]).length()
    if verbose:
        print(f"Search done in {time.time() - step_time}s"
              f"\nComplete duration: {time.time() - start_time}s")
    return longuest_snake_length, longuest_snakes


def init_prev_lengths(sheet):
    global prev_lengths
    if sheet is not None:
        prev_lengths = load_prev_lengths(sheet)
    else:
        path = "data/excel/forest_lengths_bxh.xlsx"
        if os.path.exists(path):
            workbook = load_workbook(path)
            sheet = workbook.active
            prev_lengths = load_prev_lengths(sheet)
            workbook.close()
            del sheet, workbook
        else:
            prev_lengths = [b * (_h + 1) for _h in range(h)]


def load_prev_lengths(sheet):
    prev_lengths = [sheet.cell(b + 1, _h + 1).value for _h in range(1, b)]
    prev_lengths.extend([sheet.cell(_h + b + 1, b + 1).value for _h in range(h - b + 1)])
    for _h in range(len(prev_lengths)):
        if prev_lengths[_h] is None:
            prev_lengths[_h] = b * (_h + 1)
    return prev_lengths


if __name__ == '__main__':
    low_memory_mode = False
    forest = True
    b, h = 7, 8
    graph_search((b, h), forest=forest, load=False)
    print_longuest_snake()
