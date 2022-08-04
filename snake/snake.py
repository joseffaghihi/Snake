import copy
import math
import multiprocessing as mp
import os
import random
import time
from datetime import timedelta

import numpy as np
import psutil

from snake.grid import Grid

LONGEST_SNAKES = "snakes"
LONGEST_FORESTS = "forests"
INSCRIBED_SNAKES = "inscribed_snakes"

APPLY = True
REMOVE = False


def apply_reflections(snakes):
    snakes_with_symmetry = []
    for snake in snakes:
        snakes_with_symmetry.append(copy.deepcopy(snake))
        snake.reflection(x_axis=True, y_axis=False)
        snakes_with_symmetry.append(copy.deepcopy(snake))
        snake.reflection(x_axis=False, y_axis=True)
        snakes_with_symmetry.append(copy.deepcopy(snake))
        snake.reflection(x_axis=True, y_axis=False)
        snakes_with_symmetry.append(copy.deepcopy(snake))
        if psutil.virtual_memory().percent >= 98:
            raise RuntimeError("Insufficient RAM")
    return remove_duplicates(snakes_with_symmetry)


def process_results(snake_lists, mode):
    if mode == LONGEST_SNAKES or mode == LONGEST_FORESTS:
        max_length = max(snake_lists, key=lambda s: Snake.length(s)).length()
        snake_lists = [snake for snake in snake_lists if snake.length() == max_length]
    elif mode == INSCRIBED_SNAKES:
        snake_lists = [snake for snake in snake_lists if snake.is_inscribed() is True]
    snake_lists = remove_duplicates(snake_lists)
    return snake_lists


def remove_duplicates(snakes):
    return [s for s in set(snakes)]


def remove_reflections(snakes):
    snakes_without_reflections = []
    n = len(snakes)
    for i, s in enumerate(snakes):
        if i % 100 == 0:
            print(f"\033[1K\rRemoving reflections {i}/{n}", end=" ")
        if s.is_reflection_of(snakes_without_reflections) is False:
            snakes_without_reflections.append(s)
    print(f"\033[1K\rRemoved reflections {n}/{n}")
    return snakes_without_reflections


def reflections_mp(snakes, mode, processes):
    results = []
    applied = False
    while processes > 1:
        if len(snakes) / processes >= 2000:
            random.shuffle(snakes)
            steps = [value for value in range(0, len(snakes), math.ceil(len(snakes) / processes))]
            steps.extend([len(snakes)])
            pool = mp.Pool(processes)
            if mode is REMOVE:
                [pool.apply_async(remove_reflections, args=([snakes[steps[p]:steps[p + 1]]]), callback=results.extend)
                 for p in range(len(steps) - 1)]
            elif mode is APPLY:
                [pool.apply_async(apply_reflections, args=([snakes[steps[p]:steps[p + 1]]]), callback=results.extend)
                 for p in range(len(steps) - 1)]
            pool.close()
            pool.join()
            snakes = results
            results = []
            if mode == APPLY:
                applied = True
                break
        processes = processes // 2
    if mode is REMOVE:
        results = remove_reflections(snakes)
    elif mode is APPLY:
        if applied is False:
            snakes = apply_reflections(snakes)
        results = remove_duplicates(snakes)
    return results


def side_size_for_symmetry(size) -> int:
    return int((size + size % 2) / 2)


class Snake:

    def __init__(self, grid=None, width=1, height=1):
        if grid is None:
            grid = Grid(width, height)
        self.grid = grid

    def __str__(self):
        return f"{self.grid.matrix}"

    def __gt__(self, other):
        if self.length() > other.height():
            return self
        return other

    def __eq__(self, other):
        return self.__class__ == self.__class__ and self.grid == other.grid

    def __hash__(self):
        return hash(self.grid.__hash__())

    def cell_is_snakes_end(self, i, j) -> bool:
        return self.grid.cell_is_selected(i, j) and self.number_of_selected_neighbors(i, j) <= 1

    def cell_is_valid_selection(self, i, j):
        '''DOES NOT WORK!!! CAN CREATE CYCLES'''
        if self.grid.cell_exists(i, j) is False or self.grid.cell_is_selected(i, j) \
                or self.number_of_selected_neighbors(i, j) >= 2:
            return False
        if self.grid.cell_is_selected(i - 1, j) and self.number_of_selected_neighbors(i - 1, j) == 2:
            return False
        if self.grid.cell_is_selected(i + 1, j) and self.number_of_selected_neighbors(i + 1, j) == 2:
            return False
        if self.grid.cell_is_selected(i, j - 1) and self.number_of_selected_neighbors(i, j + 1) == 2:
            return False
        if self.grid.cell_is_selected(i, j + 1) and self.number_of_selected_neighbors(i, j + 1) == 2:
            return False
        return True

    def cell_is_valid_selection_for_continuation(self, i, j):
        if self.grid.cell_exists(i, j) is False or self.grid.cell_is_selected(i, j):
            return False
        return self.number_of_selected_neighbors(i, j) == 1

    def cell_is_valid_selection_for_new_snake(self, i, j):
        if self.grid.cell_exists(i, j) is False or self.grid.cell_is_selected(i, j):
            return False
        return self.number_of_selected_neighbors(i, j) == 0

    def count_kisses(self) -> int:
        kisses = 0
        for j in range(self.grid.height - 1):
            for i in range(self.grid.width):
                if self.grid.cell_is_selected(i, j) and not self.grid.cell_is_selected(i + 1, j):
                    if self.grid.cell_is_selected(i + 1, j - 1) and not self.grid.cell_is_selected(i, j - 1):
                        kisses += 1
                    if self.grid.cell_is_selected(i + 1, j + 1) and not self.grid.cell_is_selected(i, j + 1):
                        kisses += 1
        return kisses

    def count_selected_cells_by_row(self):
        results = []
        for r in range(self.grid.height):
            n = np.count_nonzero(self.grid.matrix[::, r] == self.grid.SELECTED_VALUE)
            results.append({'b': self.grid.width, 'h': self.grid.height, 'r': r, 'n': n})
        return results

    def count_snakes(self):
        count = 0
        exploration_grid = Grid(self.grid.width, self.grid.height)
        for i in range(0, exploration_grid.width):
            for j in range(0, exploration_grid.height):
                if not (exploration_grid.cell_is_selected(i, j)) and self.cell_is_snakes_end(i, j):
                    count += 1
                    exploration_grid.select_cell(i, j)
                    next_i, next_j, prev_i, prev_j = self.find_snakes_next_part(i, j, -1, -1)
                    while next_i != -1 and next_j != -1:
                        exploration_grid.select_cell(next_i, next_j)
                        next_i, next_j, prev_i, prev_j = self.find_snakes_next_part(next_i, next_j, prev_i, prev_j)
        return count

    def count_snakes_end(self):
        count = 0
        for i in range(0, self.grid.width):
            for j in range(0, self.grid.height):
                if self.cell_is_snakes_end(i, j) is True:
                    count += 1
        return count

    def number_of_selected_neighbors(self, i, j):
        count = 0
        if self.grid.cell_is_selected(i - 1, j):
            count += 1
        if self.grid.cell_is_selected(i + 1, j):
            count += 1
        if self.grid.cell_is_selected(i, j - 1):
            count += 1
        if self.grid.cell_is_selected(i, j + 1):
            count += 1
        return count

    def find(self, mode, processes):
        print(f"Searching {mode} in {self.grid.matrix.shape}...")
        self.reset()
        cells_to_search = [(i, j) for j in range(side_size_for_symmetry(self.grid.height)) for i in
                           range(side_size_for_symmetry(self.grid.width))]
        start_time = time.time()
        process_time = 0
        finished_processes = 0
        total_processes = len(cells_to_search)
        snake_lists = []

        def update(result):
            nonlocal snake_lists, finished_processes, process_time
            snake_lists.extend(result)
            snake_lists = process_results(snake_lists, mode)
            finished_processes += 1
            progress = finished_processes / total_processes * 100
            if finished_processes <= processes:
                this_process_time = time.time() - start_time
                if process_time < this_process_time:
                    process_time = this_process_time

            if finished_processes < total_processes:
                remaining_processes = total_processes - finished_processes
                estimated_end_time = time.localtime(process_time * remaining_processes + time.time())
                print(
                    f"\033[1K\rProgress: {progress:.2f}%\tEstimated end time: {time.strftime('%x %X', estimated_end_time)}",
                    end="", flush=True)
            else:
                duration = timedelta(seconds=time.time() - start_time)
                print(f"\033[1K\rProgress: {progress:.2f}%\tDuration: {duration}", flush=True)

        pool = mp.Pool(processes)
        [pool.apply_async(self.find_process, args=(mode, i, j), callback=update) for (i, j) in cells_to_search]
        pool.close()
        pool.join()

        return process_results(snake_lists, mode)

    def find_process(self, mode, i, j):
        work_snake = copy.deepcopy(self)
        if mode == LONGEST_SNAKES or mode == INSCRIBED_SNAKES:
            work_snakes = work_snake.find_snakes_rec(i, j, mode)
        elif mode == LONGEST_FORESTS:
            work_snakes = work_snake.find_snake_forest_rec(i, j, mode)
        else:
            raise ValueError("Invalid what_to_find parameter")
        return work_snakes

    def find_snakes_rec(self, i, j, mode, level=0):
        self.select_cell(i, j)
        if self.has_possible_selections_left(i, j) is False:
            return [self]

        final_snakes = []
        if mode == INSCRIBED_SNAKES:
            final_snakes.extend([self])

        if self.cell_is_valid_selection_for_continuation(i - 1, j):
            final_snakes.extend(copy.deepcopy(self).find_snakes_rec(i - 1, j, mode, level=level + 1))
        if self.cell_is_valid_selection_for_continuation(i + 1, j):
            final_snakes.extend(copy.deepcopy(self).find_snakes_rec(i + 1, j, mode, level=level + 1))
        if self.cell_is_valid_selection_for_continuation(i, j - 1):
            final_snakes.extend(copy.deepcopy(self).find_snakes_rec(i, j - 1, mode, level=level + 1))
        if self.cell_is_valid_selection_for_continuation(i, j + 1):
            final_snakes.extend(copy.deepcopy(self).find_snakes_rec(i, j + 1, mode, level=level + 1))

        if psutil.Process(os.getpid()).memory_percent() >= 15 and (level <= 1 or psutil.virtual_memory().percent >= 95):
            if psutil.virtual_memory().percent() >= 98:
                raise RuntimeError("Insufficient RAM")
        final_snakes = process_results(final_snakes, mode)

        return final_snakes

    def find_snake_forest_rec(self, i, j, mode, level=0):
        self.select_cell(i, j)
        if self.has_other_close_snake_possibilities(i, j) is False:
            other_elligibles_cells = self.has_other_far_snake_possibilities()
            final_snakes = [self]
            for cell in other_elligibles_cells:
                if cell == (-1, -1):
                    return final_snakes
                else:
                    final_snakes.extend(copy.deepcopy(self).find_snake_forest_rec(cell[0], cell[1], mode, level + 1))

        final_snakes = [self]
        # # Selects the next cell if valid
        if self.cell_is_valid_selection_for_continuation(i - 1, j):
            final_snakes.extend(copy.deepcopy(self).find_snake_forest_rec(i - 1, j, mode, level=level + 1))
        if self.cell_is_valid_selection_for_continuation(i + 1, j):
            final_snakes.extend(copy.deepcopy(self).find_snake_forest_rec(i + 1, j, mode, level=level + 1))
        if self.cell_is_valid_selection_for_continuation(i, j - 1):
            final_snakes.extend(copy.deepcopy(self).find_snake_forest_rec(i, j - 1, mode, level=level + 1))
        if self.cell_is_valid_selection_for_continuation(i, j + 1):
            final_snakes.extend(copy.deepcopy(self).find_snake_forest_rec(i, j + 1, mode, level=level + 1))
        # Skips the next cell even if valid
        if self.cell_is_valid_selection_for_new_snake(i - 2, j):
            final_snakes.extend(copy.deepcopy(self).find_snake_forest_rec(i - 2, j, mode, level=level + 1))
        if self.cell_is_valid_selection_for_new_snake(i + 2, j):
            final_snakes.extend(copy.deepcopy(self).find_snake_forest_rec(i + 2, j, mode, level=level + 1))
        if self.cell_is_valid_selection_for_new_snake(i, j - 2):
            final_snakes.extend(copy.deepcopy(self).find_snake_forest_rec(i, j - 2, mode, level=level + 1))
        if self.cell_is_valid_selection_for_new_snake(i, j + 2):
            final_snakes.extend(copy.deepcopy(self).find_snake_forest_rec(i, j + 2, mode, level=level + 1))
        # Goes to corner cell
        if self.cell_is_valid_selection_for_new_snake(i - 1, j - 1):
            final_snakes.extend(copy.deepcopy(self).find_snake_forest_rec(i - 1, j - 1, mode, level=level + 1))
        if self.cell_is_valid_selection_for_new_snake(i + 1, j - 1):
            final_snakes.extend(copy.deepcopy(self).find_snake_forest_rec(i + 1, j - 1, mode, level=level + 1))
        if self.cell_is_valid_selection_for_new_snake(i - 1, j + 1):
            final_snakes.extend(copy.deepcopy(self).find_snake_forest_rec(i - 1, j + 1, mode, level=level + 1))
        if self.cell_is_valid_selection_for_new_snake(i + 1, j + 1):
            final_snakes.extend(copy.deepcopy(self).find_snake_forest_rec(i + 1, j + 1, mode, level=level + 1))

        if psutil.Process(os.getpid()).memory_percent() >= 15 and (level <= 1 or psutil.virtual_memory().percent >= 95):
            if psutil.virtual_memory().percent() >= 98:
                raise RuntimeError("Insufficient RAM")
            final_snakes = process_results(final_snakes, mode)

        return final_snakes

    def find_snakes_head(self):
        for i in range(0, self.grid.width):
            for j in range(0, self.grid.height):
                if self.grid.cell_is_selected(i, j) and self.number_of_selected_neighbors(i, j) <= 1:
                    return i, j
        raise RuntimeError("Snake's head not found")

    # Returns the next part i, j and the current i, j in this order
    def find_snakes_next_part(self, i, j, prev_i, prev_j):
        if self.grid.cell_is_selected(i - 1, j) and (i - 1 != prev_i or j != prev_j):
            return i - 1, j, i, j
        elif self.grid.cell_is_selected(i + 1, j) and (i + 1 != prev_i or j != prev_j):
            return i + 1, j, i, j
        elif self.grid.cell_is_selected(i, j - 1) and (i != prev_i or j - 1 != prev_j):
            return i, j - 1, i, j
        elif self.grid.cell_is_selected(i, j + 1) and (i != prev_i or j + 1 != prev_j):
            return i, j + 1, i, j
        else:
            return -1, -1, i, j

    def find_snakes_tail(self):
        for i in range(self.grid.width - 1, -1, -1):
            for j in range(self.grid.height - 1, -1, -1):
                if self.grid.cell_is_selected(i, j) and self.number_of_selected_neighbors(i, j) <= 1:
                    return i, j
        raise RuntimeError("Snake's tail not found")

    def generate_snake_io_sequence(self):
        input_sequence = []
        output_sequence = []
        sequence_grid = Grid(width=self.grid.width, height=self.grid.height)
        input_sequence.append(sequence_grid.matrix.copy())
        prev_i = -1
        prev_j = -1
        i, j = self.find_snakes_head()
        while i != -1 or j != -1:
            sequence_grid.select_cell(i, j)
            output_sequence.append(sequence_grid.matrix.copy())
            input_sequence.append(sequence_grid.matrix.copy())
            i, j, prev_i, prev_j = self.find_snakes_next_part(i, j, prev_i, prev_j)
        output_sequence.append(sequence_grid.matrix.copy())

        sequence_grid = Grid(width=self.grid.width, height=self.grid.height)
        input_sequence.append(sequence_grid.matrix.copy())
        prev_i = -1
        prev_j = -1
        i, j = self.find_snakes_tail()
        while i != -1 or j != -1:
            sequence_grid.select_cell(i, j)
            output_sequence.append(sequence_grid.matrix.copy())
            input_sequence.append(sequence_grid.matrix.copy())
            i, j, prev_i, prev_j = self.find_snakes_next_part(i, j, prev_i, prev_j)
        output_sequence.append(sequence_grid.matrix.copy())
        return input_sequence, output_sequence

    def has_possible_selections_left(self, i, j):
        return self.cell_is_valid_selection_for_continuation(i - 1, j) \
               or self.cell_is_valid_selection_for_continuation(i + 1, j) \
               or self.cell_is_valid_selection_for_continuation(i, j - 1) \
               or self.cell_is_valid_selection_for_continuation(i, j + 1)

    def has_other_close_snake_possibilities(self, i, j):
        return self.has_possible_selections_left(i, j) \
               or self.cell_is_valid_selection_for_new_snake(i - 2, j) \
               or self.cell_is_valid_selection_for_new_snake(i + 2, j) \
               or self.cell_is_valid_selection_for_new_snake(i, j - 2) \
               or self.cell_is_valid_selection_for_new_snake(i, j + 2) \
               or self.cell_is_valid_selection_for_new_snake(i - 1, j - 1) \
               or self.cell_is_valid_selection_for_new_snake(i + 1, j - 1) \
               or self.cell_is_valid_selection_for_new_snake(i - 1, j + 1) \
               or self.cell_is_valid_selection_for_new_snake(i + 1, j + 1)

    def has_other_far_snake_possibilities(self):
        other_elligible_cells = []
        for i in range(self.grid.width):
            for j in range(self.grid.height):
                if self.cell_is_valid_selection(i, j):
                    other_elligible_cells.append((i, j))
        other_elligible_cells.append((-1, -1))
        return other_elligible_cells

    def is_inscribed(self):
        top = False
        bottom = False
        left = False
        right = False
        for i in range(self.grid.width):
            if self.grid.cell_is_selected(i, 0):
                top = True
            if self.grid.cell_is_selected(i, self.grid.height - 1):
                bottom = True
        for j in range(self.grid.height):
            if self.grid.cell_is_selected(0, j):
                left = True
            if self.grid.cell_is_selected(self.grid.width - 1, j):
                right = True
        return top and bottom and left and right

    def is_reflection_of(self, other):
        for s in apply_reflections(other):
            if self.grid == s.grid:
                return True
        return False

    def length(self):
        return self.grid.count_selected_cell()

    def percentage_of_grid_filled(self):
        return self.length() / self.grid.area() * 100

    def reset(self):
        self.grid.reset()

    def reset_if_maximum_snake(self):
        number_of_snake_ends = self.count_snakes_end()
        if number_of_snake_ends <= 2 and self.grid.has_side_of_size(size=1) is False \
                or self.percentage_of_grid_filled() == 100:
            self.reset()

    def reflection(self, x_axis, y_axis):
        if x_axis:
            self.grid.reflection_x()
        if y_axis:
            self.grid.reflection_y()

    def rotate_grid(self):
        self.grid.rotate()

    def select_cell(self, i, j):
        self.grid.select_cell(i, j)

    def unselect_cell(self, i, j):
        self.grid.unselect_cell(i, j)
