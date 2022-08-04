from copy import deepcopy
from sympy import Expr, symbols, expand, sympify, pprint
from symengine import series, Matrix, eye
from xlsxwriter import Workbook
from multiprocessing import Pool, cpu_count
import time
import math
import pickle
import os

multiprocessing = True


def calculate_grid_degrees(node: list) -> None:
    # Resets all degrees to 0
    for row in node:
        for cell_index, (value, _, bottom) in enumerate(row):
            row[cell_index] = (value, 0, bottom)
    # Calculates degrees
    for row_index, row in enumerate(node):
        for cell_index, (value, degree, bottom) in enumerate(row):
            if cell_index > 0 and row[cell_index - 1][0]:
                degree += 1
            if cell_index + 1 < len(row) and row[cell_index + 1][0]:
                degree += 1
            if row_index > 0 and node[row_index - 1][cell_index][0]:
                degree += 1
                bottom = -1
            if row_index + 1 < len(node) and node[row_index + 1][cell_index][0]:
                degree += 1
            row[cell_index] = (value, degree, bottom)


def calculate_grids_degrees(nodes: list) -> None:
    for node in nodes:
        calculate_grid_degrees(node)


def calculate_row_degrees(row: list) -> None:
    # Resets all degrees to 0
    for index, (value, _, bottom) in enumerate(row):
        row[index] = (value, 0, bottom)
    # Calculates degrees
    for index, (value, degree, bottom) in enumerate(row):
        if index > 0:
            (left_value, _, _) = row[index - 1]
            if left_value:
                degree += 1
        if index + 1 < len(row):
            (right_value, _, _) = row[index + 1]
            if right_value:
                degree += 1
        if bottom is not None:
            degree += 1
        row[index] = (value, degree, bottom)


def calculate_rows_degrees(rows: list) -> None:
    for row in rows:
        calculate_row_degrees(row)


def cant_be_adjacent(index: int, current: list, added: list) -> bool:
    value, degree, other_end = current[index]
    next_val, next_deg, next_oe = added[index]
    # Prevents adding a selected cell on a cell with a degree of 2
    if next_val and degree >= 2:
        return True
    # Prevents adding a connected cell on an empty cell. Exception for isolated cells
    if next_oe is not None and not value and not (next_oe == -1 and next_deg == 1):
        return True
    # Prevents adding unconnected cell on a selected cell
    if next_val and next_oe is None and value:
        return True
    # Checks if all added row connections exist
    if segments_links_ok(index, current, added) is False:
        return True


def count_selected_cells(node: list) -> int:
    count = 0
    for cell in node:
        if cell[0] is True:
            count += 1
    return count


def create_adjacency_matrix(nodes: list) -> Matrix:
    adjacency = []
    equations = [0]
    equations.extend(find_adjacency_equations(nodes[0], nodes[1:]))
    adjacency.append(equations)
    if multiprocessing:
        pool = Pool(cpu_count())
        results = [pool.apply_async(find_adjacency_equations, args=(node, nodes)) for node in nodes]
        pool.close()
        pool.join()
        for result in results:
            adjacency.append(result.get())
    else:
        adjacency.extend([find_adjacency_equations(node, nodes) for node in nodes])
    return Matrix(adjacency)


def create_adjacency_matrix_from_scratch(b: int) -> Matrix:
    rows = gen_rows(b)
    calculate_rows_degrees(rows)
    return create_adjacency_matrix(gen_nodes(rows))


def create_adjacency_matrix_no_symbols(nodes: list) -> list:
    adjacency = []
    equations = [0]
    equations.extend(find_adjacency_equations(nodes[0], nodes[1:], with_symbols=False))
    adjacency.append(equations)
    if multiprocessing:
        pool = Pool(cpu_count())
        results = [pool.apply_async(find_adjacency_equations, args=(node, nodes, False)) for node in nodes]
        pool.close()
        pool.join()
        for result in results:
            adjacency.append(result.get())
    else:
        adjacency.extend([find_adjacency_equations(node, nodes, with_symbols=False) for node in nodes])
    return adjacency


def find_adjacency_equations(from_node: list, to: list, with_symbols=True) -> list:
    y, z = symbols("y z")
    equations = [0]
    for added in to:
        y_pow, z_pow = find_adjacency_equation(from_node, added)
        if y_pow is None:
            equations.append(0)
        elif with_symbols:
            equations.append(y ** y_pow * z ** z_pow)
        else:
            equations.append((y_pow, z_pow))
    # try:
    #     progress = ceil(to.index(from_node) / len(to) * 100)
    #     print(f"\rCreating adjacency matrix... {progress}%", end=" ")
    # except Exception:
    #     pass
    return equations


def find_adjacency_equation(current: list, added: list) -> tuple:
    y_pow, z_pow = 0, 0
    index = 0
    if has_cycle([current, added], False):
        return None, None
    while index < len(current):
        new_comp = False
        value, degree, other_end = current[index]
        next_val, next_deg, next_oe = added[index]
        if cant_be_adjacent(index, current, added):
            return None, None
        if next_val:
            y_pow += 1
            if value is False:
                new_comp = True
            if index + 1 < len(current):
                index, y_pow, z_pow = \
                    find_adjacency_equation_rec(index + 1, current, added, new_comp, y_pow, z_pow)
                if index is None:
                    return None, None
                index -= 1
            elif new_comp:
                z_pow += 1
        index += 1
    return y_pow, z_pow


# Recursion for same segment
def find_adjacency_equation_rec(index: int, current: list, added: list, new_comp: bool, y_pow: int, z_pow: int):
    if cant_be_adjacent(index, current, added):
        return None, None, None
    value, degree, other_end = current[index]
    next_val, next_deg, next_oe = added[index]
    # Check if component is new or joined
    if new_comp:
        if not next_val:
            z_pow += 1
        elif value:
            new_comp = False
    elif next_val and value:
        z_pow -= 1
    # Continuation or end of calculations
    if not next_val:
        return index + 1, y_pow, z_pow
    elif index + 1 < len(current):
        return find_adjacency_equation_rec(index + 1, current, added, new_comp, y_pow + 1, z_pow)
    elif new_comp and not value:
        z_pow += 1
    return index + 1, y_pow + 1, z_pow


# Follows the snake based on the current and past position
def follow_snake_in_rectangle(node: list, current: tuple, prev: tuple, jump_top: bool) -> (tuple, tuple):
    (row_index, cell_index) = current
    # Moving down needs to be first for segments_links_ok() to avoid errors
    if row_index > 0 and node[row_index - 1][cell_index][0] and (row_index - 1, cell_index) != prev:
        return (row_index - 1, cell_index), (row_index, cell_index)
    elif row_index + 1 < len(node) and node[row_index + 1][cell_index][0] and (row_index + 1, cell_index) != prev:
        return (row_index + 1, cell_index), (row_index, cell_index)
    elif cell_index > 0 and node[row_index][cell_index - 1][0] and (row_index, cell_index - 1) != prev:
        return (row_index, cell_index - 1), (row_index, cell_index)
    elif cell_index + 1 < len(node[row_index]) and node[row_index][cell_index + 1][0] \
            and (row_index, cell_index + 1) != prev:
        return (row_index, cell_index + 1), (row_index, cell_index)
    elif (jump_top or row_index != len(node) - 1) \
            and node[row_index][cell_index][2] is not None and node[row_index][cell_index][2] != -1 \
            and (row_index, node[row_index][cell_index][2]) != prev:
        return (row_index, node[row_index][cell_index][2]), (row_index, cell_index)
    else:
        return (-1, -1), (row_index, cell_index)


def follow_snake_in_row(row: list, current: int, prev: int) -> (int, int):
    if current > 0 and current - 1 != prev and row[current - 1][0]:
        return current - 1, current
    elif current < len(row) - 1 and current + 1 != prev and row[current + 1][0]:
        return current + 1, current
    else:
        return -1, current


def gen_nodes(rows: list) -> list:
    nodes = []
    nodes.extend(rows)
    if multiprocessing:
        pool = Pool(cpu_count())
        results = [pool.apply_async(func=gen_nodes_mp, args=(top_row, rows)) for top_row in rows]
        pool.close()
        pool.join()
        for result in results:
            nodes.extend(result.get())
    else:
        [nodes.extend(gen_nodes_mp(top_row, rows)) for top_row in rows]
    nodes = remove_useless_nodes(nodes)
    # Adds noncrrossing partitions
    nodes.extend(noncrossing_partitions(nodes))
    return remove_useless_nodes(nodes)


# Creates all possible combinations of rows
def gen_nodes_mp(top_row: list, rows_mp):
    nodes_mp = []
    for bottom_row in rows_mp:
        potential_node = [deepcopy(bottom_row), deepcopy(top_row)]
        calculate_grid_degrees(potential_node)
        # Checks if the combination is valid
        if is_ok(potential_node):
            nodes_mp.append(potential_node[1])
    return nodes_mp


def gen_nodes_from_scratch(b: int) -> list:
    rows = gen_rows(b)
    calculate_rows_degrees(rows)
    return gen_nodes(rows)


def gen_rows(b: int) -> list:
    rows = []
    row = [(False, 0, None) for _ in range(b)]
    gen_rows_rec(rows, row, 0)
    return rows


def gen_rows_rec(rows: list, row: list, i: int) -> None:
    if i == len(row):
        rows.append(row)
        return
    row0 = deepcopy(row)
    row1 = deepcopy(row)
    row1[i] = (True, 0, None)
    gen_rows_rec(rows, row0, i + 1)
    gen_rows_rec(rows, row1, i + 1)


def has_cycle(node: list, jump=True) -> bool:
    # Checks only from top line if the node only has 2 lines
    if len(node) == 2:
        row_index = 1
        for cell_index, (value, _, _) in enumerate(node[1]):
            if value and tortoise_and_hare(node, row_index, cell_index, jump):
                return True
    # Checks from all cells from all lines
    else:
        for row_index, row in enumerate(node):
            for cell_index, (value, _, _) in enumerate(row):
                if value and tortoise_and_hare(node, row_index, cell_index, jump):
                    return True
    return False


# Makes sure there is no tree and no cycle
def is_ok(node: list) -> bool:
    for row in node:
        for (selected, degree, _) in row:
            if selected and degree > 2:
                return False
    return not (has_cycle(node))


def is_same_row(row: list, other: list) -> bool:
    for (value, degree, end_index), (other_value, other_degree, other_end_index) in zip(row, other):
        if value != other_value or degree != other_degree or end_index != other_end_index:
            return False
    return True


def noncrossing_partitions(nodes: list) -> list:
    partitions = []
    if multiprocessing:
        pool = Pool(cpu_count())
        results = [pool.apply_async(noncrossing_partitions_rec, args=(node, [])) for node in nodes]
        pool.close()
        pool.join()
        for result in results:
            partitions.extend(result.get())
    else:
        [partitions.extend(noncrossing_partitions_rec(node, [])) for node in nodes]
    calculate_rows_degrees(partitions)
    return remove_useless_nodes(partitions)


def noncrossing_partitions_rec(node: list, partitions: list) -> list:
    # Ending condition: If there no possible pair left
    finished = 0
    for _, (value, degree, end_index) in enumerate(node):
        if value and degree < 2 and (end_index is None or end_index == -1):
            finished += 1
    if finished < 2:
        return partitions

    # Runs through all possible pairs
    for index, (value, degree, end_index) in enumerate(node):
        if value and degree < 2 and (end_index is None or end_index == -1):
            for other_index in range(index, len(node)):
                other_value, other_degree, other_end_index = node[other_index]
                # Checks if the pair is possible:
                # Prevents pairing same cell or neighbors cells \
                # Makes sure that other cell is eligible \
                # Prevents side by side pairing
                if index != other_index and abs(index - other_index) > 1 \
                        and other_value and other_degree < 2 and (other_end_index is None or other_end_index == -1) \
                        and (index == 0 or index > 0 and node[index - 1][2] is None) \
                        and (index == len(node) - 1 or index < len(node) - 1 and node[index + 1][2] is None) \
                        and (other_index == 0 or other_index > 0 and node[other_index - 1][2] is None) \
                        and (other_index == len(node) - 1 or
                             other_index < len(node) - 1 and node[other_index + 1][2] is None):
                    # Checks if the pair is completely linked on the row
                    ok = False
                    for i in range(min(index, other_index), max(index, other_index)):
                        if node[i][0] is False:
                            ok = True
                        elif node[i][2] is not None and node[i][2] != -1:
                            ok = False
                            break
                    if ok:  # Notes the pair and repeat
                        partition = deepcopy(node)
                        partition[index] = (value, degree, other_index)
                        partition[other_index] = (other_value, other_degree, index)
                        if creates_noncrossing_cycle(partition, index) is False:
                            partitions.append(partition)
                            noncrossing_partitions_rec(partition, partitions)
    return partitions


def creates_noncrossing_cycle(row: list, start_index: int) -> bool:
    _, _, other_end = row[start_index]
    while other_end is not None and other_end != -1:
        current_index, prev_index = follow_snake_in_row(row, other_end, -1)
        if current_index == -1:
            break
        while current_index != -1:
            current_index, prev_index = follow_snake_in_row(row, current_index, prev_index)
        if prev_index == start_index or row[prev_index][2] == start_index:
            return True
        else:
            other_end = row[prev_index][2]
    return False


def remove_duplicate_nodes(nodes: list) -> list:
    unique_nodes = []
    for node in nodes:
        if node not in unique_nodes:
            unique_nodes.append(node)
    return unique_nodes


def remove_useless_nodes(nodes: list) -> list:
    # Eliminate differences between degrees 0 and 1
    for node in nodes:
        for index, (value, degree, end_index) in enumerate(node):
            if value is False:
                node[index] = (value, 0, None)
            elif degree < 2:
                if degree == 0 and end_index is None:
                    end_index = -1
                node[index] = (value, 1, end_index)
    return remove_duplicate_nodes(nodes)


# Makes sure that all links are ok
def segments_links_ok(start_index: int, current: list, added: list) -> bool:
    goal = added[start_index][2]
    if goal is None:
        return True
    curr, prev = follow_snake_in_rectangle([current, added], (1, start_index), (-1, -1), False)
    while curr != (-1, -1):
        if curr == (1, goal):
            return True
        curr, prev = follow_snake_in_rectangle([current, added], curr, prev, False)
    # If true, means that the link is missing
    if goal != -1 or prev[0] == 1 and prev[1] != start_index and goal != prev[1]:
        return False
    return True


# Floyd's Tortoise and Hare
def tortoise_and_hare(node: list, row_index: int, cell_index: int, jump: bool):
    tortoise, tortoise_prev = follow_snake_in_rectangle(node, (row_index, cell_index), (-1, -1), jump)
    hare, hare_prev = follow_snake_in_rectangle(node, (row_index, cell_index), (-1, -1), jump)
    hare, hare_prev = follow_snake_in_rectangle(node, hare, hare_prev, jump)
    while tortoise != hare and tortoise != (-1, -1) and hare != (-1, -1):
        tortoise, tortoise_prev = follow_snake_in_rectangle(node, tortoise, tortoise_prev, jump)
        hare, hare_prev = follow_snake_in_rectangle(node, hare, hare_prev, jump)
        if hare == (-1, -1):
            break
        hare, hare_prev = follow_snake_in_rectangle(node, hare, hare_prev, jump)
        if tortoise == hare:
            return True


def load_nodes_and_matrix(b: int, with_symbols: bool):
    path = f"data/nodes_matrix/matrix_symbols={with_symbols}_{b}xh.pickle"
    if os.path.exists(path):
        with open(path, "rb") as file:
            return pickle.load(file)
    return None, None


def save_node_and_matrix(nodes: list, matrix, with_symbols) -> None:
    with open(f"data/nodes_matrix/matrix_symbols={with_symbols}_{len(nodes[0])}xh.pickle", "wb") as file:
        pickle.dump((nodes, matrix), file)


# Unfinished
# DÉVELOPPER EN SÉRIE APRÈS AVOIR ADDITIONNER LES EXPRESSIONS RATIONNELLES
def adjacency_to_serie(adjacency_matrix: Matrix, b: int, h: int) -> Expr:
    x, y, z = symbols("x y z")
    inv_mat = (eye(adjacency_matrix.rows) - x * adjacency_matrix).inv()
    sum = 0
    for j in range(2, inv_mat.rows):
        sum += inv_mat[0, j]
    serie = series(sum, x, x0=0, n=h)
    ##############################################
    # Substract uninscribed
    for in_b in range(b - 1, 0, -1):
        prev_serie = load_serie(in_b, h)
        if prev_serie is None:
            prev_serie = generate_serie(in_b, h)
        serie -= math.ceil(b / in_b) * prev_serie
    ##############################################
    return expand(serie)


def load_serie(b: int, h: int) -> Expr:
    path = f"data/series/serie{b}x{h}.pickle"
    if os.path.exists(path):
        with open(path, "rb") as file:
            return sympify(pickle.load(file))


def save_serie(serie: Expr, b: int, h: int) -> None:
    with open(f"data/series/serie{b}x{h}.pickle", "wb") as file:
        pickle.dump(serie, file)


def generate_serie(b: int, h: int) -> Expr:
    print(f"\n\tGenerating serie {b}x{h}")
    rows = gen_rows(b)
    calculate_rows_degrees(rows)
    adj_matrix = create_adjacency_matrix(gen_nodes(rows))
    serie = adjacency_to_serie(adj_matrix, b, h)
    save_serie(serie, b, h)
    return serie


###########################################################
# TESTS AND VERIFICATIONS
def excel_export(nodes, adj_matrix, workbook=None):
    close = False
    if workbook is None:
        close = True
        workbook = Workbook(f"data/excel/AdjacencyMatrix_{len(nodes[0])}xh.xlsx")
    sheet = workbook.add_worksheet(f"{len(nodes[0])}xh")
    sheet.write(0, 1, f"{nodes[0]}")
    sheet.write(1, 0, f"{nodes[0]}")
    for n, node in enumerate(nodes):
        sheet.write(0, n + 2, f"{node}")
        sheet.write(n + 2, 0, f"{node}")
    for i in range(adj_matrix.cols):
        for j in range(adj_matrix.rows):
            sheet.write(i + 1, j + 1, f"{adj_matrix[i, j]}")
    if close:
        workbook.close()


def test_node(b, test_i):
    rows = gen_rows(b)
    calculate_rows_degrees(rows)
    nodes = gen_nodes(rows)
    find_adjacency_equations(nodes[test_i], nodes)


##############################################################


if __name__ == '__main__':
    # b = 3
    h = 10
    multiprocessing = True

    # workbook = Workbook("data/excel/AdjacencyMatrix_bxh.xlsx")
    for b in range(9, 10):
        print(f"\nFor b = {b + 1}")
        start_time = time.time()
        print("Generating basic rows...", end=" ")
        rows = gen_rows(b + 1)
        calculate_rows_degrees(rows)
        print(f"Done in {(time.time() - start_time):.3f}s\nGenerating nodes...", end=" ")
        start_time = time.time()
        nodes = gen_nodes(rows)
        del rows
        print(f"Done in {(time.time() - start_time):.3f}s\nCreating adjacency matrix...", end=" ")
        start_time = time.time()
        adj_matrix = create_adjacency_matrix(nodes)
        # print(f"Done in {(time.time() - start_time):.3f}s\nExporting to Excel...", end=" ")
        # start_time = time.time()
        # excel_export(nodes, adj_matrix, workbook)
        # print(f"Done in {(time.time() - start_time):.3f}s\nComputing serie from adjacency matrix...", end=" ")
        # workbook.close()

        # print(f"Computing serie from adjacency matrix...", end=" ")
        # start_time = time.time()
        # serie = adjacency_to_serie(adj_matrix, b, h)
        print(f"Done in {(time.time() - start_time):.3f}s\n")
        # for node in nodes:
        #     print(node)
        # pprint(adj_matrix, wrap_line=False)
        # pprint(serie, wrap_line=False)
        # save_serie(serie, b, h)
    # excel_export(nodes, adj_matrix)

    # test_node(3, -1)
