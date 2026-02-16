import pygame
import math
import neat
import numpy as np
import pickle
import os
import random

pygame.init()

pickle_in = open("data/winner.pickle", "rb")
model = pickle.load(pickle_in)

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, "config-feedforward.txt")
config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
net = neat.nn.FeedForwardNetwork.create(model, config)

window_size = (800, 900)
box_size = 50
GRID_SHAPE = (math.floor(window_size[0]/box_size), math.floor((window_size[1]-100)/box_size))
grid = np.array([[False]*(GRID_SHAPE[0])]*(GRID_SHAPE[1]))
pygame.display.set_caption("Very Hungry Snek")
window = pygame.display.set_mode(window_size)
run = True
PREV_KEY = None
FPS = 5

SNEK_HEAD = pygame.transform.scale(pygame.image.load("data/snek_head.png"), (math.floor(box_size-(box_size/10)), math.floor(box_size-(box_size/10))))
SNEK_TOUNG = pygame.transform.scale(pygame.image.load("data/snek_toung.png"), (math.floor(box_size-(box_size/10)), math.floor(box_size-(box_size/10))))

def get_rand_pos(snake):
    valid_pos = False
    while not valid_pos:
        pos = [random.randint(1, GRID_SHAPE[0] - 2), random.randint(1, GRID_SHAPE[1] - 2)]
        if not pos in snake:
            valid_pos = True
    return pos

snake = []
new_snake = []

snake.append([8, 8])
apple = get_rand_pos(snake)


def text_objects(colour, message, font):
    text_surface = font.render(message, False, colour)
    text_rectangle = text_surface.get_rect()
    return text_surface, text_rectangle


def draw(snakes, apples):
    window.fill((0, 0, 0))
    #  Draw grid
    for y in range(GRID_SHAPE[0]):
        for x in range(GRID_SHAPE[1]):
            pygame.draw.rect(window, (255, 255, 255), (x * box_size, (y * box_size)+100, box_size - math.floor(box_size / 10), box_size - math.floor(box_size / 10)))
    try:
        #  Apple
        pygame.draw.rect(window, (255, 0, 0), (apples[0][0] * box_size, (apples[0][1] * box_size)+100, box_size - math.floor(box_size / 10), box_size - math.floor(box_size / 10)))
        #  Snake
        s = 0
        for body in snakes[0]:
            rect = (body[0] * box_size, (body[1] * box_size)+100, box_size - math.floor(box_size / 10), box_size - math.floor(box_size / 10))
            if s == 0:
                pygame.draw.rect(window, (0, 120, 6), rect)
            else:
                pygame.draw.rect(window, (0, 166, 6), rect)
    except: pass

    pygame.display.update()


def get_input(button1, button2, prev_dir):

    opposite_direction = {0:1, 1:0, 2:3, 3:2}
    if button1 != opposite_direction[prev_dir]:
        return button1
    else:
        return button2


def update(snake, apple, direction):
    global new_snake
    eaten = False
    score = 0
    game_over = False

    if snake[0] == apple:
        apple = get_rand_pos(snake)
        score += 5
        eaten = True

    #  Update head
    if direction == 0:
        new_snake.append([snake[0][0], snake[0][1] + 1])
    elif direction == 1:
        new_snake.append([snake[0][0], snake[0][1] - 1])
    elif direction == 2:
        new_snake.append([snake[0][0] - 1, snake[0][1]])
    else:
        new_snake.append([snake[0][0] + 1, snake[0][1]])

    for i in range(len(snake)):
        if i != 0:
            new_snake.append(snake[i - 1])

    # Make sure snake is allowed to follow its tail
    body_to_check = snake if eaten else snake[:-1]
    if new_snake[0] in body_to_check:
        game_over = True

    if eaten:
        new_snake.append(snake[len(snake)-1])

    snake.clear()
    for i in range(len(new_snake)):
        snake.append(new_snake[i])
    new_snake.clear()


    if snake[0][0] < 0 or snake[0][1] < 0 or\
       snake[0][0] > GRID_SHAPE[0]-1 or snake[0][1] > GRID_SHAPE[1]-1:
        game_over = True
        score -= 30

    return snake, apple, game_over, score



def distance_food(snake, apple):
    d_food_x = snake[0][0]-apple[0]
    d_food_y = snake[0][1]-apple[1]
    return d_food_x, d_food_y
def disance_foot_tot(snake, apple):
    d_food_x, d_food_y = distance_food(snake, apple)
    tot_dist = math.sqrt((d_food_y**2)+(d_food_x**2))
    return tot_dist
def distance_wall(snake):
    n_wall = snake[0][1]
    s_wall = 15 - snake[0][1]
    e_wall = 15 - snake[0][0]
    w_wall = snake[0][0]
    return n_wall, s_wall, e_wall, w_wall
def distance_tail(snake):
    tail_x = snake[0][0] - snake[len(snake)-1][0]
    tail_y = snake[0][1] - snake[len(snake)-1][1]
    return tail_x, tail_y
def surrounding(snake):
    north = 0
    south = 0
    east = 0
    west = 0
    if [snake[0][0], snake[0][1]-1] in snake:
        north = 1
    if [snake[0][0], snake[0][1]+1] in snake:
        south = 1
    if [snake[0][0]+1, snake[0][1]] in snake:
        east = 1
    if [snake[0][0]-1, snake[0][1]] in snake:
        west = 1
    return north, south, east, west

def get_data(snake, apple, current_dir):

    d_food_x, d_food_y = distance_food(snake, apple)
    n_wall, s_wall, e_wall, w_wall = distance_wall(snake)
    tail_x, tail_y = distance_tail(snake)
    north, south, east, west = surrounding(snake)

    data = np.array([d_food_x, d_food_y,
                    n_wall, s_wall, e_wall, w_wall,
                    tail_x, tail_y,
                    north, south, east, west,
                    current_dir])

    return data


clock = pygame.time.Clock()
current_dir = 0
while run:
    clock.tick(FPS)

    output = np.array(net.activate(get_data(snake, apple, current_dir)))

    first = int(np.argmax(output))
    second = int(np.argsort(output)[-2])

    direction = get_input(first, second, current_dir)
    current_dir = direction

    snake, apple, game_over, score = update(snake, apple, direction)

    draw([snake], [apple])

    if game_over:
        run = False

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            pygame.quit()
            quit()