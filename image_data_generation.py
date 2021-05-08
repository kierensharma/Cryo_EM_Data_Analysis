import glob
import cv2
import numpy as np
import pandas as pd
from troponin import get_mrc_image
from troponin import find_actin_ll
from troponin import get_trp_coords
from troponin import grab_troponin
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt
from tqdm import tqdm

def troponin_generator():
    datagen = ImageDataGenerator(
            rotation_range=90,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0,
            zoom_range=0,
            horizontal_flip=True,
            fill_mode='reflect')

    troponin_files = []
    for file in glob.glob("/Volumes/KierenSSD/University/Troponin_Images/*.png"):
        troponin_files.append(file)

    for troponin_file in tqdm(troponin_files):
        img = load_img(troponin_file)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape) 

        i = 0
        for batch in datagen.flow(x, batch_size=1,
                                save_to_dir='/Volumes/KierenSSD/University/Generated_troponin_dataset_new', save_prefix='troponin', save_format='jpeg'):
            i += 1
            if i > 20:
                break  

def not_troponin_generator():
    num = 0
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

        for y in range(150, 3750, 150):
            for x in range(150, 3750, 150):
                check = 'false'
                for coords in data:
                    x_coord = coords[0]
                    y_coord = coords[1]
                    if (x-75) <= x_coord <= (x+75) and (y-75) <= y_coord <= (y+75):
                        check = 'true'
                
                if check == 'false':
                    not_troponin_img = grab_troponin(image, x, y, grab_rad=150)
                    not_trop_img_path = '/Volumes/KierenSSD/University/Not_Troponin_Images/not_trop_' + str(num) + '.png'
                    plt.imsave(not_trop_img_path, not_troponin_img)
                    num += 1

def make_dataset():
    x = []
    y = []
    
    not_trop_files = []
    for file in glob.glob("/Volumes/KierenSSD/University/Not_Troponin_Images/*.png"):
        not_trop_files.append(file)
    trop_files = []
    for file in glob.glob("/Volumes/KierenSSD/University/Generated_troponin_dataset/*.jpeg"):
        trop_files.append(file)
    
    for not_trop_file in tqdm(not_trop_files[:10000]):
        img = cv2.imread(not_trop_file)
        scale_percent = 50 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        flattened_image = image.flatten()
        flattened_image = flattened_image.tolist()
        # print(len(flattened_image))
        if len(flattened_image) != 67500:
            continue

        x.append(flattened_image)
        y.append(0)

    for trop_file in tqdm(trop_files):
        img = cv2.imread(trop_file)
        scale_percent = 50 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        flattened_image = image.flatten()
        flattened_image = flattened_image.tolist()
        if len(flattened_image) != 67500:
            continue

        x.append(flattened_image)
        y.append(1)

    return np.array(x), np.array(y)

def main():
    # troponin_generator()
    # not_troponin_generator()
    x, y = make_dataset()
    print(x)
    print(x.shape)
    print(type(x))
    print(y)
    print(y.shape)
    print(type(y))

if __name__ == '__main__':
    main()
