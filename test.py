import cv2
from troponin import get_mrc_image
from troponin import find_actin_ll
from troponin import get_trp_coords
from troponin import grab_troponin
import matplotlib.pyplot as plt
from itertools import chain 

image_path = '/Volumes/KierenSSD/University/Thinfil_micrographs_and_coords/micrographs_final/FoilHole_24671225_Data_24671816_24671817_20181025_2332-80407.mrc'
image = get_mrc_image(image_path)
imgplot = plt.imshow(image)

line_image, line_image_s, long_lines = find_actin_ll(image)
imgplot = plt.imshow(line_image)
plt.show()

labels, data = get_trp_coords('/Volumes/KierenSSD/University/Thinfil_micrographs_and_coords/coordinates/FoilHole_24671225_Data_24671816_24671817_20181025_2332-80407.star')

for coords in chain(data):
    x_coord = coords[0]
    y_coord = coords[1]

    # plt.scatter(x_coord, y_coord, c='white', marker="1")

    troponin_img = grab_troponin(image, x_coord, y_coord, grab_rad=300)
    imgplot = plt.imshow(troponin_img)
    # plt.show()

# plt.show()
