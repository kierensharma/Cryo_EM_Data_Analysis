import glob
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
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
    mrc_files = []
    for file in glob.glob("/Volumes/KierenSSD/University/Thinfil_micrographs_and_coords/micrographs_final/*.mrc"):
        mrc_files.append(file)
    star_files = []
    for file in glob.glob("/Volumes/KierenSSD/University/Thinfil_micrographs_and_coords/coordinates/*.star"):
        star_files.append(file)

    mrc_files.sort()
    star_files.sort()

    

def main():
    troponin_generator()
    # not_troponin_generator()

if __name__ == '__main__':
    main()
