import unittest

from snake.snake import Snake, LONGEST_FORESTS


# TEST SNAKE 1
# length = 17, snake_count = 1, kisses = 0
def snake1():
    snake = Snake(width=5, height=5)
    snake.select_cell(0, 0)
    snake.select_cell(0, 1)
    snake.select_cell(0, 2)
    snake.select_cell(0, 3)
    snake.select_cell(0, 4)
    snake.select_cell(1, 0)
    snake.select_cell(1, 4)
    snake.select_cell(2, 0)
    snake.select_cell(2, 2)
    snake.select_cell(2, 3)
    snake.select_cell(2, 4)
    snake.select_cell(3, 0)
    snake.select_cell(4, 0)
    snake.select_cell(4, 1)
    snake.select_cell(4, 2)
    snake.select_cell(4, 3)
    snake.select_cell(4, 4)
    return snake


# TEST SNAKE 2
# length = 17, snake_count = 1, kisses = 2
def snake2():
    snake = Snake(width=5, height=5)
    snake.select_cell(0, 1)
    snake.select_cell(0, 2)
    snake.select_cell(0, 3)
    snake.select_cell(0, 4)
    snake.select_cell(1, 0)
    snake.select_cell(1, 1)
    snake.select_cell(1, 4)
    snake.select_cell(2, 0)
    snake.select_cell(2, 2)
    snake.select_cell(2, 3)
    snake.select_cell(2, 4)
    snake.select_cell(3, 0)
    snake.select_cell(3, 1)
    snake.select_cell(4, 1)
    snake.select_cell(4, 2)
    snake.select_cell(4, 3)
    snake.select_cell(4, 4)
    return snake


# TEST FOREST 1
# length = 21, snake_count = 3, kisses = 6
def forest1():
    snake = Snake(width=5, height=6)
    snake.select_cell(0, 0)
    snake.select_cell(0, 1)
    snake.select_cell(0, 3)
    snake.select_cell(0, 4)
    snake.select_cell(0, 5)
    snake.select_cell(1, 0)
    snake.select_cell(1, 2)
    snake.select_cell(1, 3)
    snake.select_cell(1, 5)
    snake.select_cell(2, 1)
    snake.select_cell(2, 2)
    snake.select_cell(2, 4)
    snake.select_cell(2, 5)
    snake.select_cell(3, 0)
    snake.select_cell(3, 3)
    snake.select_cell(3, 4)
    snake.select_cell(4, 0)
    snake.select_cell(4, 1)
    snake.select_cell(4, 2)
    snake.select_cell(4, 3)
    snake.select_cell(4, 5)
    return snake


# TEST FOREST 2
# Special case to see if all maximum forests can be reached
def forest2():
    snake = Snake(width=4, height=8)
    snake.select_cell(0, 0)
    snake.select_cell(0, 2)
    snake.select_cell(0, 3)
    snake.select_cell(0, 5)
    snake.select_cell(0, 6)
    snake.select_cell(0, 7)
    snake.select_cell(1, 0)
    snake.select_cell(1, 2)
    snake.select_cell(1, 4)
    snake.select_cell(1, 5)
    snake.select_cell(1, 7)
    snake.select_cell(2, 0)
    snake.select_cell(2, 3)
    snake.select_cell(2, 4)
    snake.select_cell(2, 6)
    snake.select_cell(2, 7)
    snake.select_cell(3, 0)
    snake.select_cell(3, 1)
    snake.select_cell(3, 2)
    snake.select_cell(3, 3)
    snake.select_cell(3, 5)
    snake.select_cell(3, 6)
    return snake


class SnakeTests(unittest.TestCase):

    def test_count_kisses(self):
        self.assertEqual(0, snake1().count_kisses())
        self.assertEqual(2, snake2().count_kisses())
        self.assertEqual(6, forest1().count_kisses())

    def test_count_snakes(self):
        self.assertEqual(1, snake1().count_snakes())
        self.assertEqual(1, snake1().count_snakes())
        self.assertEqual(3, forest1().count_snakes())

    # Special case to see if all maximum forests can be reached
    def test_find_snake_forests_rec(self):
        partial_forest = forest2()
        partial_forest.unselect_cell(0, 2)
        partial_forest.unselect_cell(0, 3)
        partial_forest.unselect_cell(1, 2)
        forests = partial_forest.find_snake_forest_rec(0, 0, LONGEST_FORESTS)

        forest2_is_present = False
        for forest in forests:
            if forest == forest2():
                forest2_is_present = True
                break
        self.assertTrue(forest2_is_present)


if __name__ == '__main__':
    unittest.main()
