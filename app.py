import pygame
import math
import neat
import numpy as np
import pickle
import os
import random

pygame.init()

window_size = (800, 900)
box_size = 50
GRID_SHAPE = (math.floor(window_size[0]/box_size), math.floor((window_size[1]-100)/box_size))
grid = np.array([[False]*(GRID_SHAPE[0])]*(GRID_SHAPE[1]))
pygame.display.set_caption("Very Hungry Snek")
window = pygame.display.set_mode(window_size)
run = True
# DISPLAY to see the training in progress
DISPLAY = False
# SLOW to slow down the visualization of training
SLOW = False
PREV_KEY = None
# True to save best ever model, not the final model
SAVE_BEST = True
FPS = 5
generations = 1500

eat_self_penalty = 0
starve_penalty = 0
leave_grid_penalty = 0
eat_apple_bonus = 2

SNEK_HEAD = pygame.transform.scale(pygame.image.load("data/snek_head.png"), (math.floor(box_size-(box_size/10)), math.floor(box_size-(box_size/10))))
SNEK_TOUNG = pygame.transform.scale(pygame.image.load("data/snek_toung.png"), (math.floor(box_size-(box_size/10)), math.floor(box_size-(box_size/10))))


new_snake = []

generation = 1


def get_rand_pos(snake, seed, seed_step):
    valid_pos = False
    while not valid_pos:
        random.seed(seed + 6358*seed_step)
        pos = [random.randint(1, GRID_SHAPE[0] - 2), random.randint(1, GRID_SHAPE[1] - 2)]
        if not pos in snake:
            valid_pos = True
        else:
            seed += 63427654
    return pos


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


def human_input():
    global SLOW
    global PREV_KEY
    global DISPLAY
    pressed = pygame.key.get_pressed()

    if PREV_KEY != pressed:
        if pressed[pygame.K_s]:
            if SLOW:
                SLOW = False
            else:
                SLOW = True
        elif pressed[pygame.K_d]:
            if DISPLAY:
                DISPLAY = False
            else:
                DISPLAY = True

    PREV_KEY = pressed


def update(snake, apple, direction, seed, seed_step):
    global new_snake
    eaten = False
    score = 0
    game_over = False


    if snake[0] == apple:
        apple = get_rand_pos(snake, seed, seed_step)
        score += eat_apple_bonus
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
        score += eat_self_penalty

    if eaten:
        new_snake.append(snake[len(snake)-1])

    snake.clear()
    for i in range(len(new_snake)):
        snake.append(new_snake[i])
    new_snake.clear()


    if snake[0][0] < 0 or snake[0][1] < 0 or\
       snake[0][0] > GRID_SHAPE[0]-1 or snake[0][1] > GRID_SHAPE[1]-1:
        game_over = True
        score += leave_grid_penalty

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
    if [snake[0][0], snake[0][1]-1] in snake or snake[0][1]-1 < 0:
        north = 1
    if [snake[0][0], snake[0][1]+1] in snake or snake[0][1]+1 > 15:
        south = 1
    if [snake[0][0]+1, snake[0][1]] in snake or snake[0][0]+1 > 15:
        east = 1
    if [snake[0][0]-1, snake[0][1]] in snake or snake[0][0]-1 < 0:
        west = 1
    return north, south, east, west

def get_data(snake, apple, current_dir):

    d_food_x, d_food_y = distance_food(snake, apple)
    n_wall, s_wall, e_wall, w_wall = distance_wall(snake)
    #tail_x, tail_y = distance_tail(snake)
    north, south, east, west = surrounding(snake)

    data = np.array([d_food_x, d_food_y,
                    #n_wall, s_wall, e_wall, w_wall,
                    #tail_x, tail_y,
                    north, south, east, west,
                    current_dir])
    '''
    data = np.zeros(shape=GRID_SHAPE)
    for y in range(GRID_SHAPE[1]):
        for x in range(GRID_SHAPE[0]):
            if apple == [x, y]:
                data[x][y] = 1
            elif snake[0] == [x, y]:
                data[x][y] = -0.5
            elif [x, y] in snake:
                data[x][y] = -1
    '''



    return data.flatten()


avg_fitness = []
max_fitness = []
high_scores = []
def main(genomes, config):
    global run
    global DISPLAY
    global SLOW
    global FPS
    global high_scores
    global max_fitness
    global avg_fitness
    global fitness
    nets = []
    ge = []

    fitnesses = []

    snakes = []
    current_dir = []
    hunger = []
    apples = []
    seed_step = 0

    total_fit = 0
    high_score = 0

    clock = pygame.time.Clock()

    #  Set up networks
    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        snakes.append([[8, 8], [8, 7]])
        current_dir.append(0)
        hunger.append(200)
        g.fitness = 0
        ge.append(g)

    seed = random.randint(0, 1098652)
    for i in range(len(snakes)):
        apples.append(get_rand_pos(snakes[i], seed, seed_step))

    while run and len(snakes) > 0:
        if SLOW:
            clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
                break

        i = 0
        while i < len(snakes):
            snake = snakes[i]

            output = np.array(nets[i].activate(get_data(snake, apples[i], current_dir[i])))
            fisrt = int(np.argmax(output))
            second = int(np.argsort(output)[-2])

            direction = get_input(fisrt, second, current_dir[i])
            current_dir[i] = direction

            snake, apples[i], game_over, score = update(snake, apples[i], direction, seed, seed_step)
            if score < eat_apple_bonus:
                hunger[i] -= 1
            else:
                hunger[i] = 200

            if len(snake) > high_score:
                high_score = len(snake)

            #score += math.floor(5/(1+disance_foot_tot(snake, apples[i])))
            score += 0.01
            ge[i].fitness += score
            total_fit += score

            if hunger[i] <= 0:
                ge[i].fitness += starve_penalty

            if game_over or hunger[i] <= 0:
                fitnesses.append(ge[i].fitness)

                snakes.pop(i)
                nets.pop(i)
                apples.pop(i)
                ge.pop(i)
                current_dir.pop(i)
                hunger.pop(i)
                continue
            i += 1

        if DISPLAY:
            draw(snakes, apples)
        human_input()
        seed_step += 1


    highest = 0
    for data in fitnesses:
        if data > highest:
            highest = data
    avg_fitness.append(round(total_fit/generations, 2))
    max_fitness.append(highest)
    high_scores.append(high_score)

    if len(avg_fitness) == generations:
        np.save("data/AvgPIX", np.array(avg_fitness))
        np.save("data/MaxPIX", np.array(max_fitness))
        np.save("data/LengthPIX", np.array(high_scores))


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    final = p.run(main, generations)
    best_ever = max(stats.most_fit_genomes, key=lambda g: g.fitness)

    with open("data/final.pickle", "wb") as f:
        pickle.dump(final, f)
    with open("data/best.pickle", "wb") as f:
        pickle.dump(best_ever, f)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)
