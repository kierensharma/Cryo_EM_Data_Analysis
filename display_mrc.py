from troponin import get_mrc_image
import matplotlib.pyplot as plt

pathname = '/Volumes/KierenSSD/University/Thinfil_micrographs_and_coords/micrographs_final/FoilHole_24671225_Data_24671816_24671817_20181025_2332-80407.mrc'
image = get_mrc_image(pathname)
imgplot = plt.imshow(image)
plt.show()