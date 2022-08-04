import numpy as np
from openpyxl import Workbook

from snake.snake import remove_reflections
from snake.snake_exporter import snakes_to_xlsx


def main():
    filepath = '../data/excel/3banc.xlsx'
    workbook = Workbook()
    workbook.remove(workbook.active)
    b = 3
    for h in range(3, 31, 3):
        print(f'snake_forest_{b}x{h}')
        reflections = False
        try:
            snakes = np.load(f'../data/npy/snake_forest_{b}x{h}_no-reflections.npy', allow_pickle=True)
        except FileNotFoundError:
            reflections = True
            snakes = np.load(f'../data/npy/snake_forest_{b}x{h}.npy', allow_pickle=True)
        snakes_with_benches = []
        for s, snake in enumerate(snakes):
            print(f'\rAnalysing snake {s}/{snakes.shape[0]}', end='')
            left_bench, right_bench = 0, 0
            left_n, right_n = 0, 0
            left_previous_degree2, right_previous_degree2 = False, False
            for n in range(h):
                # LEFT SIDE
                if n == left_n:
                    left_n += 1
                    if snake.grid.cell_is_selected(0, n):
                        if snake.grid.cell_is_selected(1, n) and snake.grid.cell_is_selected(0, n + 1) \
                                and snake.grid.cell_is_selected(0, n + 2) and snake.grid.cell_is_selected(1, n + 2):
                            left_bench += 1
                            left_n = n + 3
                            left_previous_degree2 = False
                        elif snake.grid.cell_degree(0, n) == 2:
                            if left_previous_degree2:
                                left_bench = 2
                            left_previous_degree2 = True
                        else:
                            left_previous_degree2 = False
                    else:
                        left_previous_degree2 = False
                else:
                    left_previous_degree2 = False
                # RIGHT SIDE
                if n == right_n:
                    right_n += 1
                    if snake.grid.cell_is_selected(b - 1, n):
                        if snake.grid.cell_is_selected(b - 2, n) and snake.grid.cell_is_selected(b - 1, n + 1) \
                                and snake.grid.cell_is_selected(b - 1, n + 2) \
                                and snake.grid.cell_is_selected(b - 2, n + 2):
                            right_bench += 1
                            right_n = n + 3
                            right_previous_degree2 = False
                        elif snake.grid.cell_degree(b - 1, n) == 2:
                            if right_previous_degree2:
                                right_bench = 2
                            right_previous_degree2 = True
                        else:
                            right_previous_degree2 = False
                    else:
                        right_previous_degree2 = False
                else:
                    right_previous_degree2 = False
            # SINGLE BENCH?
            if left_bench == 1 or right_bench == 1:
                snakes_with_benches.append(snake)
        print(f'\rAnalysing snake {snakes.shape[0]}/{snakes.shape[0]}')
        if len(snakes_with_benches) > 0:
            if reflections:
                snakes_with_benches = remove_reflections(snakes_with_benches)
            snakes_to_xlsx(snakes_with_benches, filepath, workbook.create_sheet(f'{b}x{h}'))
    workbook.save(filepath)


if __name__ == '__main__':
    main()
