import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def data_generator(directory):
    datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

    troponin_directory = os.listdir(directory)
    # troponin_directory.sort()

    for troponin in troponin_directory:
        img_path = '/Volumes/KierenSSD/University/Troponin_Images/' + troponin
        img = load_img(img_path)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape) 

        i = 0
        for batch in datagen.flow(x, batch_size=1,
                                save_to_dir='/Volumes/KierenSSD/University/Generated_troponin_dataset', save_prefix='troponin', save_format='jpeg'):
            i += 1
            if i > 20:
                break  

def main():
    data_generator('/Volumes/KierenSSD/University/Troponin_Images/')

if __name__ == '__main__':
    main()
