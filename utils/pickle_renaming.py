import glob
import pickle

from numpy import save
from numpy.lib.format import read_magic, _check_version, _read_array_header


class RenamingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "snake":
            module = "snake.snake"
        elif module == "grid":
            module = "snake.grid"
        return super().find_class(module, name)


def rename_pickle(path):
    with open(path, 'rb') as fp:
        version = read_magic(fp)
        _check_version(version)
        dtype = _read_array_header(fp, version)[2]
        assert dtype.hasobject
        snakes = RenamingUnpickler(fp).load()
    save(path, snakes)


if __name__ == "__main__":
    paths = glob.glob("data/npy/*")
    for path in paths:
        print(f"Renaming pickles for {path}")
        rename_pickle(path)
