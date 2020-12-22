import scaffan.image as scim
import matplotlib.pyplot as plt
import sed3
import numpy as np
from skimage.color import rgb2hsv

NDPI_EXAMPLE = "D:\\FAV\\Scaffold\\Scaffan-analysis\\PIG-002_J-18-0091_HE.ndpi"
CZI_EXAMPLE = "D:\\ML-Data\\Anicka - reticular fibers\\J9_9\\J9_9_a.czi"

custom_grad = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
]


def get_crop(img, center=None, size=(100, 100)):
    if center is None:
        center = img.shape[0] // 2, img.shape[1] // 2

    return img[center[0] - size[0] // 2: center[0] + size[0] // 2, center[1] - size[1] // 2: center[1] + size[1] // 2,
           :]


def anotate_texture(img):
    ed = sed3.sed3(img)
    ed.show()

    seeds = ed.seeds
    seeds = seeds[:, :, 0]
    # print(img[seeds == 1])
    seeds = seeds.astype('int8')

    return seeds


def load_data():
    img = np.load('image.npy')
    img = img / 255.0
    seeds = np.load('seeds.npy')
    return img, seeds


def get_masks_from_seeds(seeds):
    levels = set(seeds)


def get_centroid_colors_rgb(img, seeds):
    # Get all types of annotation
    levels = set(list(seeds.reshape(seeds.size)))
    levels.remove(0)
    levels = list(levels)
    centroids = []

    for level in levels:
        centroids.append(np.sum(img[seeds == level], axis=0) / np.sum(seeds == level))

    return centroids


def find_centroids():
    img, seeds = load_data()
    centroids_rgb = get_centroid_colors_rgb(img, seeds)
    centroids_hsv = [hue_to_continuous_2d(rgb2hsv(pixel.reshape(1, 1, 3))) for pixel in centroids_rgb]

    return centroids_hsv


def closest_neighbor():
    pass


def hue_to_continuous_2d(img):
    """Takes hsv image and returns ´hhsv´ format, where hue is replaced by 2 values - sin(hue) and cos(hue)"""
    hue = np.expand_dims(img[:, :, 0], -1)

    hue_x = np.cos(hue * 2 * np.pi)
    hue_y = np.sin(hue * 2 * np.pi)

    img = np.concatenate((hue_x, hue_y, img[:, :, 1:]), axis=-1)

    return img


def get_image_filter(img, seeds):
    img = rgb2hsv(img)
    img = hue_to_continuous_2d(img)

    centroids_hsv = find_centroids()

    centroid_maps = []

    for centroid in centroids_hsv:
        centroid_map = np.tile(centroid, img.shape[0] * img.shape[1]).reshape(img.shape)
        centroid_maps.append(centroid_map)

    centroid_distances = []

    for centroid_map in centroid_maps:
        centroid_distances.append(abs(img - centroid_map))

    centroid_distances = np.sum(centroid_distances, axis=-1)

    filter = np.argmin(np.array(centroid_distances), axis=0)

    return filter


def load_example_img():
    # get full img as numpy array
    anim = scim.AnnotatedImage(CZI_EXAMPLE)
    img = anim.get_full_view().get_region_image()
    img = get_crop(img, (8700, 12000), (1400, 1400))
    return img


def create_annotation():
    img = load_example_img()
    seeds = anotate_texture(img)
    np.save('seeds.npy', seeds)
    np.save('image.npy', img)


def filter_image():
    img, seeds = load_data()

    plt.figure()
    plt.imshow(img)
    plt.title("Original image")

    fltr = get_image_filter(img, seeds)

    filters = []

    plt.figure()
    plt.imshow(fltr, cmap='brg')
    plt.title("3 different tissue - RGB")

    # Black
    plt.figure()
    plt.imshow(img * np.stack([fltr == 0, fltr == 0, fltr == 0], axis=-1) + np.stack([fltr != 0, fltr != 0, fltr != 0],
                                                                                     axis=-1).astype('int32'))
    plt.title("Black tissue")

    # White
    plt.figure()
    plt.imshow(img * np.stack([fltr == 1, fltr == 1, fltr == 1], axis=-1))
    plt.title("White tissue")

    # Brown
    plt.figure()
    plt.imshow(img * np.stack([fltr == 2, fltr == 2, fltr == 2], axis=-1))
    plt.title("Brown tissue")

    plt.show()


if __name__ == '__main__':
    # create_annotation()
    filter_image()
