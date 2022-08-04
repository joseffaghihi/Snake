import numpy as np


class Grid:
    SELECTED_VALUE = 1
    UNSELECTED_VALUE = -1

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.matrix = np.ones((width, height)) * Grid.UNSELECTED_VALUE

    def __eq__(self, other):
        return np.array_equal(self.matrix, other.matrix)

    def __hash__(self):
        return hash(self.matrix.tostring())

    def __str__(self):
        return f"{self.matrix}"

    def area(self):
        return self.width * self.height

    def cell_degree(self, i, j):
        degree = 0
        if self.cell_is_selected(i - 1, j):
            degree += 1
        if self.cell_is_selected(i + 1, j):
            degree += 1
        if self.cell_is_selected(i, j - 1):
            degree += 1
        if self.cell_is_selected(i, j + 1):
            degree += 1
        return degree

    def cell_exists(self, i, j):
        return 0 <= i < self.width and 0 <= j < self.height

    def cell_is_selected(self, i, j):
        return self.cell_exists(i, j) and self.matrix[i, j] == Grid.SELECTED_VALUE

    def count_selected_cell(self):
        number_of_selected_cell = 0
        for i in range(0, self.width):
            for j in range(0, self.height):
                if self.matrix[i, j] == Grid.SELECTED_VALUE:
                    number_of_selected_cell += 1
        return number_of_selected_cell

    def has_side_of_size(self, size):
        return self.width == size or self.height == size

    def reflection_x(self):
        self.matrix = self.matrix[::-1]

    def reflection_y(self):
        self.matrix = self.matrix[::, ::-1]

    def reset(self):
        self.matrix = np.ones((self.width, self.height)) * Grid.UNSELECTED_VALUE

    def rotate(self):
        self.matrix = self.matrix.T
        prev_width = self.width
        self.width = self.height
        self.height = prev_width

    def select_cell(self, i, j):
        self.matrix[i, j] = Grid.SELECTED_VALUE

    def unselect_cell(self, i, j):
        self.matrix[i, j] = Grid.UNSELECTED_VALUE
