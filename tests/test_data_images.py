import unittest
import numpy as np
from gan.data_images import *


class DataImagesTests(unittest.TestCase):
    def test_mean_cell_standard_deviation(self):
        b, h = 5, 8
        snakes = np.load(f'../data/npy/snake_{b}x{h}.npy', allow_pickle=True)
        snake_image = snake_to_image(snakes[0], (31, 37))
        pixels = np.array(snake_image, dtype=int).T
        mean_std = mean_cell_standard_deviation(pixels, (b, h))
        self.assertEqual(0, mean_std, msg=f'Mean standard deviation is {mean_std} but is supposed to be 0')

    def test_detect_cell_dims(self):
        b_expected, h_expected = 7, 13
        success = 0
        snakes = np.load(f'../data/npy/snake_{b_expected}x{h_expected}.npy', allow_pickle=True)
        for snake in snakes:
            image_dims = np.random.randint(39, 40, size=2)
            image = snake_to_image(snake, tuple(image_dims))
            pixels = np.array(image, dtype=int).T
            detected_dims = detect_cell_dims(pixels)
            if detected_dims[0] == b_expected and detected_dims[1] == h_expected:
                success += 1
            else:  # FOR ANALYSIS PURPOSES
                print(f'Expected {(b_expected, h_expected)} and detected {detected_dims} from image {image_dims}')
                image.save(f'Expected {(b_expected, h_expected)} Detected {detected_dims}.png')
        success_rate = success / len(snakes) * 100
        self.assertEqual(100, success_rate, msg=f"Success rate is only {success_rate}%")

    def test_rect_dims_from_noise_image(self):
        success_rate = 0
        img_dims = (20, 35)
        rect_dims_array = np.random.randint(1, 10, size=(100, 2))
        for rect_dims in rect_dims_array:
            noise = generate_noise_image(rect_dims, img_dims)
            b, h = get_rect_dims_from_noise_image(noise)
            try:
                if rect_dims[0] == b and rect_dims[1] == h:
                    success_rate += 1
            except AssertionError:
                noise.save(f"expected_{rect_dims} but got {b, h}.png")
        self.assertEqual(100, success_rate, msg=f"Success rate is only {success_rate}%")

    def test_snake_to_image(self):
        b, h = 4, 9
        image_b, image_h = 31, 24
        b_array = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3]
        h_array = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 8, 8]

        snakes = np.load(f"../data/npy/snake_{b}x{h}.npy", allow_pickle=True)
        snake_image = snake_to_image(snakes[0], (image_b, image_h))
        pixels = snake_image.load()
        for i, b in enumerate(b_array):
            for j, h in enumerate(h_array):
                self.assertEqual(snakes[0].grid.cell_is_selected(b, h), pixels[i, j] == 255,
                                 msg="Snake to image conversion has failed")


if __name__ == '__main__':
    unittest.main()
