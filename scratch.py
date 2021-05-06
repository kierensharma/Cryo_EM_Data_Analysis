import cv2
from troponin import get_mrc_image
from troponin import find_actin_ll
from troponin import get_trp_coords
from troponin import grab_troponin
import matplotlib.pyplot as plt

image_path = '/Volumes/KierenSSD/University/Thinfil_micrographs_and_coords/micrographs_final/FoilHole_24671225_Data_24671816_24671817_20181025_2332-80407.mrc'
image = get_mrc_image(image_path)
imgplot = plt.imshow(image)
# plt.imsave('test.png', image)
# plt.show()

# line_image, line_image_s, long_lines = find_actin_ll(image)
# imgplot = plt.imshow(line_image)
# plt.show()

labels, data = get_trp_coords('/Volumes/KierenSSD/University/Thinfil_micrographs_and_coords/coordinates/FoilHole_24671225_Data_24671816_24671817_20181025_2332-80407.star')
fig, ax = plt.subplots(nrows=23, ncols=23)

for y, row in zip(range(150, 3750, 150), ax):
    for x, col in zip(range(150, 3750, 150), row):
        check = 'false'
        for coords in data:
            x_coord = coords[0]
            y_coord = coords[1]
            if (x-75) <= x_coord <= (x+75) and (y-75) <= y_coord <= (y+75):
                check = 'true'
        
        if check == 'false':
            troponin_img = grab_troponin(image, x, y, grab_rad=150)
            imgplot = col.imshow(troponin_img)
            col.axis('off')

plt.subplots_adjust(wspace=0, hspace=0)
plt.show()

# for coords in data_fake:
#     x_coord = coords[0]
#     y_coord = coords[1]

#     # plt.scatter(x_coord, y_coord, c='white', marker="1")

#     troponin_img = grab_troponin(image, x_coord, y_coord, grab_rad=150)
#     imgplot = plt.imshow(troponin_img)
#     plt.show()
#     # plt.imsave('trop.png', troponin_img)

# # plt.show()
