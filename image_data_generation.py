import glob
import cv2
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

def make_dataframe():
    image_dataframe = pd.DataFrame (columns = ['Label','Image'])
    # image = cv2.imread("/Volumes/KierenSSD/University/Troponin_Images/trop_215.png")
    # flattened_image = image.flatten()
    # new_data = {'Label':1, 'Image':flattened_image}
    # image_dataframe = image_dataframe.append(new_data, ignore_index=True)
    # print(image_dataframe)
    
    not_trop_files = []
    for file in glob.glob("/Volumes/KierenSSD/University/Not_Troponin_Images/*.png"):
        not_trop_files.append(file)
    trop_files = []
    for file in glob.glob("/Volumes/KierenSSD/University/Generated_troponin_dataset/*.png"):
        trop_files.append(file)
    
    for not_trop_file in tqdm(not_trop_files):
        image = cv2.imread(not_trop_file)
        flattened_image = image.flatten()
        new_data = {'Label':0, 'Image':flattened_image}
        image_dataframe = image_dataframe.append(new_data, ignore_index=True)

    for trop_file in tqdm(trop_files):
        image = cv2.imread(trop_file)
        flattened_image = image.flatten()
        new_data = {'Label':1, 'Image':flattened_image}
        image_dataframe = image_dataframe.append(new_data, ignore_index=True)

    image_dataframe = image_dataframe.sample(frac = 1)
    image_dataframe.to_csv('image_dataframe.csv')

def main():
    # troponin_generator()
    # not_troponin_generator()
    make_dataframe()

if __name__ == '__main__':
    main()
