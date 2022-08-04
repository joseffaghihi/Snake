import argparse
import os.path as path
from multiprocessing import cpu_count

from numpy import save, load

from snake.snake import Snake, reflections_mp, LONGEST_SNAKES, INSCRIBED_SNAKES, REMOVE, APPLY
from snake.snake_exporter import snakes_to_xlsx


def do_exportation(filename, snakes, export):
    save(f"data/npy/{filename}.npy", snakes)
    if export is True:
        filepath = f"data/excel/{filename}.xlsx"
        print(f"Exporting as {filepath}...", end=" ")
        snakes_to_xlsx(snakes, filepath)
        print("Done")


def find(grid_width, grid_height, mode, processes, export_as_xlsx=False, reflections=False, redo=False):
    if mode == LONGEST_SNAKES:
        filename = f"snake_{grid_width}x{grid_height}"
    elif mode == INSCRIBED_SNAKES:
        filename = f"inscribed_snakes_{grid_width}x{grid_height}"
    else:
        raise ValueError("Invalid mode")

    if reflections is False:
        filename = filename + '_no-reflections'

    loaded = False
    if redo is False and path.exists(f"data/npy/{filename}.npy"):
        snakes = load(f"data/npy/{filename}.npy", allow_pickle=True)
        loaded = True
    else:
        base_snake = Snake(width=grid_width, height=grid_height)
        snakes = base_snake.find(mode, processes)
        if reflections:
            snakes = reflections_mp(snakes, APPLY, processes)
        else:
            snakes = reflections_mp(snakes, REMOVE, 1)

    print(f"Résultats {filename}")
    if mode == LONGEST_SNAKES:
        print(f"\tMax length = {snakes[0].length()}")
    print(f"\tNumber of possibilities = {len(snakes)}")
    if loaded is False:
        do_exportation(filename, snakes, export_as_xlsx)
    return snakes


if __name__ == "__main__":
    def argparser():
        parser = argparse.ArgumentParser(description="Exhaustive search for snakes in a grid")
        parser.add_argument("mode", choices=[LONGEST_SNAKES, INSCRIBED_SNAKES],
                            help="specifies what to search")
        parser.add_argument("width", metavar='W', type=int, help="width of the grid")
        parser.add_argument("height", metavar='H', type=int, help="height of the grid")
        parser.add_argument("-l", "--loop", action="store_true", help="runs all search from 1x1 to WxH")
        parser.add_argument("-e", "--excel_export", action="store_true", help="exports results as .xlsx file")
        parser.add_argument("--processes", metavar='n', default=cpu_count(), type=int,
                            help="number of processes launch to search (defaults to the number of CPUs)")
        parser.add_argument("--redo", action="store_true",
                            help="redo calculation even if the corresponding .npy file exists")
        parser.add_argument("--reflections", action="store_true", help="include all reflections")
        return parser.parse_args()


    def main():
        args = argparser()
        width = min(args.width, args.height)
        height = max(args.width, args.height)
        if args.loop:
            for i in range(2, width + 1):
                for j in range(i, height + 1):
                    search(args, i, j)
                    print()
        else:
            search(args, width, height)


    def search(args, i, j):
        find(i, j, args.mode, args.processes, export_as_xlsx=args.excel_export, reflections=args.reflections,
             redo=args.redo)


    main()
