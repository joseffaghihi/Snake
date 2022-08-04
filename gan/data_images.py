import numpy as np
import tensorflow as tf
from PIL import Image

from snake.snake import Snake, LONGEST_SNAKES

MINIMUM_SAMPLE_SIZE_FOR_EACH_DIMENSIONS = 0  # 100


def get_rect_dims_from_noise_image(image: Image):
    b, h = 1, 1
    pixels = image.load()
    prev_gray_val = pixels[0, 0]
    for i in range(image.size[0]):
        if pixels[i, 0] != prev_gray_val:
            prev_gray_val = pixels[i, 0]
            b += 1
    prev_gray_val = pixels[0, 0]
    for j in range(image.size[1]):
        if pixels[0, j] != prev_gray_val:
            prev_gray_val = pixels[0, j]
            h += 1
    return b, h


def generate_noise_one_hot_data(rect_dims: tuple, image_dims: tuple):
    img = generate_noise_with_image_dims(rect_dims, image_dims)
    return tf.one_hot(np.asarray(img), depth=256)


def generate_noise_with_image_dims(rect_dims: tuple, image_dims: tuple) -> np.ndarray:
    grid = np.random.choice(np.arange(256, dtype=np.uint8), size=(rect_dims[0], rect_dims[1]),
                            replace=False)  # FULL GRAYSCALE NO DUPLICATES
    # grid = np.random.randint(256, size=(rect_dims[0], rect_dims[1]), dtype=np.uint8)  # FULL GRAYSCALE
    # grid = np.random.choice(np.array([0, 255], dtype=np.uint8), size=(rect_dims[0], rect_dims[1]))  # BLACK OR WHITE ONLY
    rect_b, rect_h = rect_dims[0], rect_dims[1]
    i_arrays = np.array_split(range(image_dims[0]), rect_b)
    j_arrays = np.array_split(range(image_dims[1]), rect_h)
    noise = np.zeros(image_dims, dtype=np.uint8)
    for b in range(rect_b):
        for h in range(rect_h):
            for i in i_arrays[b]:
                for j in j_arrays[h]:
                    noise[i, j] = grid[b, h]
    return noise


def generate_noise_dict(min_rect_dims, max_rect_dims, image_dims) -> dict:
    noise_dict = {}
    for b in range(min_rect_dims[0], max_rect_dims[0] + 1):
        for h in range(min_rect_dims[1], max_rect_dims[1] + 1):
            noise_dict[f'({b}, {h})'] = generate_noise_with_image_dims((b, h), image_dims)
    return noise_dict


def snake_to_image(snake: Snake, image_dims: tuple) -> Image:
    snake_b, snake_h = snake.grid.width, snake.grid.height
    i_arrays = np.array_split(range(image_dims[0]), snake_b)
    j_arrays = np.array_split(range(image_dims[1]), snake_h)
    img = Image.new('L', image_dims)
    pixels = img.load()
    for b in range(snake_b):
        for h in range(snake_h):
            for i in i_arrays[b]:
                for j in j_arrays[h]:
                    if snake.grid.cell_is_selected(b, h):
                        pixels[i, j] = 255
                    else:
                        pixels[i, j] = 0
    return img


def mean_cell_standard_deviation(array_image, real_dims):
    i_arrays = np.array_split(range(array_image.shape[0]), real_dims[0])
    j_arrays = np.array_split(range(array_image.shape[1]), real_dims[1])
    stds = []
    for b in range(real_dims[0]):
        for h in range(real_dims[1]):
            pixels = []
            for i in i_arrays[b]:
                for j in j_arrays[h]:
                    pixels.append(array_image[i, j])
            stds.append(np.std(pixels))
    return np.mean(stds)


def image_to_snake(image: Image, rect_dims: tuple) -> Snake:
    snake = Snake(width=rect_dims[0], height=rect_dims[1])
    i_arrays = np.array_split(range(image.size[0]), rect_dims[0])
    j_arrays = np.array_split(range(image.size[1]), rect_dims[1])
    image = image.load()
    for b in range(rect_dims[0]):
        for h in range(rect_dims[1]):
            pixels = []
            for i in i_arrays[b]:
                for j in j_arrays[h]:
                    pixels.append(image[i, j])
            if np.mean(pixels) >= 127:
                snake.select_cell(b, h)
    return snake


def binary_to_image(encoded) -> Image:
    img = Image.new('1', (encoded.shape[0], encoded.shape[1]))
    pixels = img.load()
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            pixels[i, j] = int(encoded[i][j])
    return img


def one_hots_to_image(encoded) -> Image:
    img = Image.new('L', (encoded.shape[0], encoded.shape[1]))
    pixels = img.load()
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            pixels[i, j] = int(tf.argmax(encoded[i][j]).numpy())
    return img


def one_hots_to_array_img(encoded) -> np.ndarray:
    pixels = np.zeros(encoded.shape[0:2])
    for i in range(pixels.shape[0]):
        for j in range(pixels.shape[1]):
            pixels[i, j] = int(tf.argmax(encoded[i][j]))
    return pixels


def load_varied_dims_data(min_rect_dims: tuple, max_rect_dims: tuple, image_dims: tuple, data_type: str):
    images = []
    dims = []
    print('Training dataset:')
    for b in range(min_rect_dims[0], max_rect_dims[0] + 1):
        for h in range(min_rect_dims[1], max_rect_dims[1] + 1):
            data = load_real_data((b, h), image_dims, data_type)
            print(f'{data.shape[1]} in {b}x{h}')
            for sample in data[0]:
                images.append(sample)
                dims.append([b, h])
    return np.asarray([images, dims], dtype=object)


def load_square_dims_data(min_square_dim: int, max_square_dim: int, image_dims: tuple, data_type: str):
    images = []
    dims = []
    print('Training dataset:')
    for c in range(min_square_dim, max_square_dim + 1):
        data = load_real_data((c, c), image_dims, data_type)
        print(f'{data.shape[1]} in {c}x{c}')
        for sample in data[0]:
            images.append(sample)
            dims.append([c, c])
    return np.asarray([images, dims], dtype=object)


def load_real_data(rect_dims: tuple, image_dims: tuple, data_type: str):
    b, h = rect_dims
    if b > h:
        b = h
        h = b
    if data_type == "MAX_SNAKES":
        snakes = np.load(f"data/npy/snake_{b}x{h}.npy", allow_pickle=True)
    elif data_type == "MAX_FOREST":
        snakes = np.load(f"data/npy/snake_forest_{b}x{h}.npy", allow_pickle=True)
    else:
        raise ValueError("Invalid data_type argument")

    if rect_dims[0] > rect_dims[1]:
        for snake in snakes:
            snake.rotate_grid()

    if len(snakes) < MINIMUM_SAMPLE_SIZE_FOR_EACH_DIMENSIONS:
        snakes = np.resize(snakes, MINIMUM_SAMPLE_SIZE_FOR_EACH_DIMENSIONS)

    data = np.array([[np.array(snake_to_image(snake, image_dims), dtype=int).T for snake in snakes]])
    return tf.one_hot(data, depth=256)


def detect_cell_dims(pixels: np.ndarray) -> (int, int):
    h_cell = []
    for i in range(pixels.shape[0]):
        previous_pixel = None
        edges_pos = [0]
        for j in range(pixels.shape[1]):
            if previous_pixel is not None and abs(pixels[i, j] - previous_pixel) > 127:
                edges_pos.append(j)
            previous_pixel = pixels[i, j]
        edges_pos.append(pixels.shape[1])
        for n in range(len(edges_pos) - 1):
            h_cell.append(edges_pos[n + 1] - edges_pos[n])

    b_cell = []
    for j in range(pixels.shape[1]):
        previous_pixel = None
        edges_pos = [0]
        for i in range(pixels.shape[0]):
            if previous_pixel is not None and abs(pixels[i, j] - previous_pixel) > 127:
                edges_pos.append(i)
            previous_pixel = pixels[i, j]
        edges_pos.append(pixels.shape[0])
        for n in range(len(edges_pos) - 1):
            b_cell.append(edges_pos[n + 1] - edges_pos[n])

    def calculate_weighted_cell_dim(cells_dims_array):
        cells_dims_array.sort()
        bincount = np.bincount(cells_dims_array)

        def argmax_with_neighbors(bincount):
            best_val, best_val_score = 0, 0
            for val, n_appearances in enumerate(bincount):
                if n_appearances != 0 and val != len(bincount) - 1:
                    val_score = n_appearances + bincount[val - 1] + bincount[val + 1]
                    if val_score > best_val_score:
                        best_val, best_val_score = val, val_score
            return best_val

        most_frequent = argmax_with_neighbors(bincount)
        bin_indices = [most_frequent - 1, most_frequent, most_frequent + 1]
        number = bincount[bin_indices[0]] + bincount[bin_indices[1]] + bincount[bin_indices[2]]
        return (bin_indices[0] * bincount[bin_indices[0]] + bin_indices[1] * bincount[bin_indices[1]] + bin_indices[2] *
                bincount[bin_indices[2]]) / number

    b_cell_dim = calculate_weighted_cell_dim(b_cell)
    h_cell_dim = calculate_weighted_cell_dim(h_cell)
    b = round(pixels.shape[0] / b_cell_dim)
    h = round(pixels.shape[1] / h_cell_dim)
    return b, h


# TESTS
def save_real_png(rect_dims, img_dims):
    snake = Snake(width=rect_dims[0], height=rect_dims[1])
    snake = snake.find(mode=LONGEST_SNAKES, processes=4)[0]
    snake_image = snake_to_image(snake, img_dims)
    snake_image.save("real.png")


def save_noise_png(rect_dims, img_dims):
    Image.fromarray(generate_noise_with_image_dims(rect_dims, img_dims), mode='L').save("noise.png")


if __name__ == '__main__':
    img_dims = 100, 100
    rect_dims = 5, 5
    # save_real_png(rect_dims, img_dims)
    save_noise_png(rect_dims, img_dims)
    # test_rect_dims_from_noise_image(img_dims)
