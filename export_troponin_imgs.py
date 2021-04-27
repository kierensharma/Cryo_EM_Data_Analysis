import os
import cv2
from troponin import get_mrc_image
from troponin import find_actin_ll
from troponin import get_trp_coords
from troponin import grab_troponin
import matplotlib.pyplot as plt
from itertools import chain 

def get_trop_imgs():
    num = 0
    mrc_direct = os.listdir('/Volumes/KierenSSD/University/Thinfil_micrographs_and_coords/micrographs_final')
    mrc_direct.sort()
    star_direct = os.listdir('/Volumes/KierenSSD/University/Thinfil_micrographs_and_coords/coordinates')
    star_direct.sort()
    for mrc_file, star_file in zip(mrc_direct, star_direct):
        mrc_path = '/Volumes/KierenSSD/University/Thinfil_micrographs_and_coords/micrographs_final/' + mrc_file[2:]
        image = get_mrc_image(mrc_path)
        star_path = '/Volumes/KierenSSD/University/Thinfil_micrographs_and_coords/coordinates/' + star_file[2:]
        labels, data = get_trp_coords(star_path)

        for coords in data:
            x_coord = coords[0]
            y_coord = coords[1]

            troponin_img = grab_troponin(image, x_coord, y_coord, grab_rad=150)
            trop_img_path = '/Volumes/KierenSSD/University/Troponin_Images/trop_' + str(num) + '.png'
            plt.imsave(trop_img_path, troponin_img) 

            num += 1

        num += 1

def main():
    get_trop_imgs()

if __name__ == '__main__':
    main()
