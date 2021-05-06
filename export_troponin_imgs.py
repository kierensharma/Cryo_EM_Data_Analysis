import glob
import cv2
from troponin import get_mrc_image
from troponin import find_actin_ll
from troponin import get_trp_coords
from troponin import grab_troponin
import matplotlib.pyplot as plt
from itertools import chain 
from tqdm import tqdm

def get_trop_imgs():
    mrc_files = []
    for file in glob.glob("/Volumes/KierenSSD/University/Thinfil_micrographs_and_coords/micrographs_final/*.mrc"):
        mrc_files.append(file)
    star_files = []
    for file in glob.glob("/Volumes/KierenSSD/University/Thinfil_micrographs_and_coords/coordinates/*.star"):
        star_files.append(file)

    mrc_files.sort()
    star_files.sort()

    for mrc_file, star_file in zip(mrc_files, star_files):
        image = get_mrc_image(mrc_file)
        labels, data = get_trp_coords(star_file)

        for coords in data:
            x_coord = coords[0]
            y_coord = coords[1]

            troponin_img = grab_troponin(image, x_coord, y_coord, grab_rad=150)
            trop_img_path = '/Volumes/KierenSSD/University/New_Troponin_Images/trop_' + str(num) + '.png'
            plt.imsave(trop_img_path, troponin_img)

            num += 1

        num += 1

def main():
    get_trop_imgs()

if __name__ == '__main__':
    main()
