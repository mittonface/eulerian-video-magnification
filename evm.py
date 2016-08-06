import numpy as np
import cv2
from spacial import generate_gaussian_pyramid, generate_laplacian_pyramid

def main():
    test_image = cv2.imread('images/bird.jpg')

    # visualize gaussian pyramid
    gauss_pyramid = generate_gaussian_pyramid(test_image, 5)

    count = 0
    for level in gauss_pyramid:
        cv2.imwrite("images/g%d.jpg" % count, level)
        count += 1

    laplace_pyramid = generate_laplacian_pyramid(test_image, 5)

    count=0
    for level in laplace_pyramid:
        cv2.imwrite("images/l%d.jpg" % count, level)
        count += 1

if __name__ == "__main__":
    main()