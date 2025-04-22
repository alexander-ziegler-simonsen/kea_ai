#!/usr/bin/env python

import os
import math
import random
import subprocess
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


x = [1, 3, 4.5, 5.5]
y = [2.5, 3, 3, 3.5]

print('x', x)
print('y', y)

def h(theta, x):
    return theta[0] + theta[1] * x


def gradient_step(theta, x, y, alpha, verbose=True, howOften = 100):
    if verbose: print("Gradient step ", theta, x, y, alpha)
    delta = np.zeros(np.shape(theta))
    m = len(y)
    for i in range(m):
        delta[0] -= (2/float(m)) * (y[i] - h(theta, x[i]))
        delta[1] -= (2/float(m)) * (y[i] - h(theta, x[i])) * x[i]
        if verbose: print(i, delta)
    if (verbose and (i % howOften == 0)):
        print("Theta", theta - alpha * delta)
        print("Cost", sum(1/(2*m) * np.square(h(theta, np.array(x)) - np.array(y))))
    return theta - alpha * delta


def gradient_descent(x, y, initial_theta, alpha, iterations, verbose=True, howOften=100):
    theta_history = []
    theta = initial_theta
    for i in range(iterations):
        if (verbose and (i % howOften == 0)): print("** Iteration ", i)
        theta = gradient_step(theta, x, y, alpha, verbose, howOften)
        theta_history.append(theta)
    return theta, theta_history

# changed this value from 1200 --------------------------------------------
theta, theta_history = gradient_descent(x, y, np.array([0,0]), 0.01, 500, True, 100)

print('-----------------------------theta', theta)

xmin, xmax = min(x), max(x)
ymin, ymax = min(y), max(y)


# clean up output directory
img_path = 'img/'
if not os.path.exists(img_path):
    os.makedirs(img_path)
img_files = os.listdir(img_path)
for img_file in img_files:
    img_file_path = os.path.join(img_path, img_file)
    if os.path.isfile(img_file_path):
        os.remove(img_file_path)


print('-----------------------------Creating image files ...')
for i, theta in enumerate(tqdm(theta_history)):
    plt.scatter(x, y)
    plt.xlim(math.floor(xmin), math.ceil(xmax))
    plt.ylim(math.floor(ymin), math.ceil(ymax))
    a = np.linspace(xmin, xmax, 2)
    b = theta[0] + a * theta[1]
    plt.plot(a, b)
    plt.title(f'Iterations:{i:004}')
    plt.savefig(f'{img_path}{i:004}.png')
    plt.close()


print('-----------------------------Creating image palette ...')
ffmpeg_command_create_palette = [
    'ffmpeg',
    '-y',
    '-i', f'{img_path}%04d.png',
    '-vf', 'palettegen',
    'palette.png',
]
subprocess.call(ffmpeg_command_create_palette)


print('-----------------------------Crating animated gif file ...')
ffmpeg_command = [
    'ffmpeg',
    '-y',
    '-i', f'{img_path}%04d.png',
    '-i', 'palette.png',
    '-filter_complex', 'fps=60,scale=600:-1:flags=lanczos[x];[x] [1:v]paletteuse',
    'gdanim.gif',
]
subprocess.call(ffmpeg_command)

print('-----------------------------Done!')
