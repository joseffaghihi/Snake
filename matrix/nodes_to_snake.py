from snake.snake import Snake


def nodes_to_snake(nodes: list) -> Snake:
    snake = Snake(width=len(nodes[0]), height=len(nodes))
    for i, node in enumerate(nodes):
        for j, cell in enumerate(node):
            if cell[0]:
                snake.select_cell(j, i)
    return snake


def node_snakes_to_snakes(node_snakes: list) -> list:
    return [nodes_to_snake(node_snake) for node_snake in node_snakes]
