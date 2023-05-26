import noise
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import cm

shape = (256, 256)
scale = 100
octaves = 10
persistence = 0.618
lacunarity = 1.618


def perline_noise():
    a = np.zeros(shape)
    seed = np.random.randint(0, 100)
    r = np.random.randint(100, 200)
    print(seed, r)
    for i in range(shape[0]):
        for j in range(shape[1]):
            a[i][j] = noise.pnoise2(i / scale,
                                    j / scale,
                                    octaves=octaves,
                                    persistence=persistence,
                                    lacunarity=lacunarity,
                                    repeatx=r,
                                    repeaty=r,
                                    base=seed)

    a = (a - np.min(a)) / np.ptp(a)
    img = Image.fromarray(np.uint8(cm.gist_earth(a) * 255))
    print(img)
    return img


def random_noise(shape):
    random_image = Image.fromarray(
        np.random.randint(0, 255, shape, dtype=np.dtype('uint8'))
    )
    return random_image


def add_color(a):
    blue = [65, 105, 225]
    green = [34, 139, 34]
    beach = [238, 214, 175]
    array = np.copy(a)
    shape = (100, 100)
    color_array = np.zeros(array.shape + (3,))
    for i in range(shape[0]):
        for j in range(shape[1]):
            if array[i][j] < -0.05:
                color_array[i][j] = blue
            elif array[i][j] < 0:
                color_array[i][j] = beach
            elif array[i][j] < 1.0:
                color_array[i][j] = green

    array_display = np.copy(color_array)
    array_display = array_display.astype(np.uint8)

    noise_image = Image.fromarray(array_display, mode='RGB')
    plt.imshow(noise_image)
    plt.show()
    return array


def show_image(img):
    plt.imshow(img)
    plt.show()
    print('img')


show_image(perline_noise())
