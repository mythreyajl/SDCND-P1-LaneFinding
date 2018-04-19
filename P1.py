#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
import sys

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

# Helpers
import P1_helpers


# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

    # Blur input image
    img = image
    img = P1_helpers.gaussian_blur(img, 3)

    # Apply Binomial Threshold
    low = np.array([200, 200, 200])
    up = np.array([255, 255, 255])
    mask = cv2.inRange(img, low, up)
    masked_image = cv2.bitwise_and(img, img, mask=mask)

    # Canny Detection
    edges = P1_helpers.canny(masked_image, 50, 150)

    # Hough transform
    lines = P1_helpers.hough_lines(edges, 1, np.pi/180, 30, 40, 150)

    # Mask Image
    shape = lines.shape
    w = shape[1]
    h = shape[0]
    bl = np.array([w * 0.10, h * 1.00])
    tl = np.array([w * 0.35, h * 0.6])
    tr = np.array([w * 0.55, h * 0.6])
    br = np.array([w * 0.95, h * 1.00])
    vertices = np.array([bl, tl, tr, br])
    cropped_image = P1_helpers.region_of_interest(lines, np.array([vertices], dtype=np.int32))

    # Overlay and return result
    overlaid_image = P1_helpers.weighted_img(image, cropped_image, 0.5, 1.0)

    plt.subplot(231)
    plt.imshow(image)
    plt.subplot(232)
    plt.imshow(img)
    plt.subplot(233)
    plt.imshow(edges)
    plt.subplot(234)
    plt.imshow(lines)
    plt.subplot(235)
    plt.imshow(cropped_image)
    plt.subplot(236)
    plt.imshow(overlaid_image)

    return overlaid_image


if __name__ == '__main__':
    os.listdir("test_images/")

    #reading in an image
    image = mpimg.imread('test_images/solidWhiteRight.jpg')

    #printing out some stats and plotting
    print('This image is:', type(image), 'with dimensions:', image.shape)
    # plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')

    white_output = 'test_videos_output/solidWhiteRight.mp4'
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
    clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)

    HTML("""<video width="960" height="540" controls> <source src="{0}"> </video>""".format(white_output))