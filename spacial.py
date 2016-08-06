import cv2
import numpy as np

"""
The first step in the process is "spacial decomposition". This means that
we create a pyramid of images where each level of the pyramid captures different
aspects
"""

def generate_gaussian_pyramid(image, levels):
    """
    Generates a gaussian pyramid. Essentially, the bottom layer of the pyramid is the original
    image. The next layer is that image, but half the size. The resulting pixels in each
    subsequent layer is the result of convolving a guassian kernel, hence the name.

    :param image: The image at the base of the pyramid.
    :param levels: Max depth of the pyramid
    :return: A list where each layer is a different level of the pyramid.
    """


    base_layer = np.empty_like(image, dtype=np.float32)
    base_layer = image[:]

    pyramid = [base_layer]

    for i in range(1, levels):
        base_layer = cv2.pyrDown(base_layer)
        pyramid.append(base_layer)

    return pyramid

def generate_laplacian_pyramid(image, levels):
    """
    Generate a Lapplacian pyramid. Each level of the laplacian pyramid contains information
    from different spatial frequencies in the image. The idea here is to use a guassian pyramid.
    Smaller layers of the gaussian pyramid encode coarser details about the overall image. By
    subtracting layers of the gaussian pyramid we get layers that capture data from each frequency

    :param image:
    :param levels:
    :return:
    """

    gaussian_pyramid = generate_gaussian_pyramid(image, levels)

    pyramid = []

    # first we subtract the second layer of the gaussian from the 1st layer. The result
    # is the first layer of our laplacian pyramid. Next we subtract 3rd from second. etc.
    # Of course we need to scale the smaller layer up to make subtraction work.
    for i in range(levels-1):
        layer = np.subtract(gaussian_pyramid[i], cv2.pyrUp(gaussian_pyramid[i+1])
        pyramid.append(layer)

    return pyramid