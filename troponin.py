import cv2
import numpy as np

from skimage import exposure
from skimage.util import invert

import mrcfile

def find_actin(img, kern_size=5, sigma=5, low_th=30, high_th=80, rh=2, tht=180, threshld=150,
				min_ll=50, max_lg=20, thickness=5, erosion_reps=10):
    # convert to image to single-channel
    if len(img.shape)>2:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = img.astype(np.uint8)
    img_inv = invert(img)


    # smooth with gaussian blur
    kernel_size = kern_size
    img_inv = cv2.GaussianBlur(img_inv,(kernel_size, kernel_size),sigma)
    
    # edge detection using Canny algorithm
    low_threshold = low_th
    high_threshold = high_th
    edges = cv2.Canny(img_inv, low_threshold, high_threshold)

    # run Hough lines on edge image
    rho = rh  # distance resolution in pixels of the Hough grid
    theta = np.pi / tht  # angular resolution in radians of the Hough grid
    threshold = threshld  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = min_ll  # minimum number of pixels making up a line
    max_line_gap = max_lg  # maximum gap in pixels between connectable line segments

    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    
    # draw lines onto blank image
    line_image = np.copy(img) * 0
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),255,thickness)
    
    # thicken lines into a searchable area
    thick_lines = np.copy(line_image)
    thick_lines = invert(thick_lines)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))    # structuring Element

    # erode the negative space around line image from Hough stage
    for i in range(erosion_reps):
        erode = cv2.erode(thick_lines,kernel)
        thick_lines = erode.copy()

    thick_lines = invert(thick_lines)
    
    percent = np.count_nonzero(thick_lines)/thick_lines.size     # percent coverage of highlighted area
    thick_lines_image = cv2.addWeighted(img, 1, thick_lines, 0.25, 0)  # superimpose highlighted area over image
    
    return thick_lines, thick_lines_image, percent, edges

def find_actin_ll(img, kern_size=5, sigma=0, low_th=30, high_th=80, rh=2, tht=180, threshld=150,
				min_ll=50, max_lg=20, true_min_ll=150, thickness=5):
    # convert to image to single-channel
    if len(img.shape)>2:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = img.astype(np.uint8)
    img_inv = invert(img)

    # smooth with gaussian blur
    kernel_size = kern_size
    img_inv = cv2.GaussianBlur(img_inv,(kernel_size, kernel_size),sigma)
    
    # edge detection using Canny algorithm
    low_threshold = low_th
    high_threshold = high_th
    edges = cv2.Canny(img_inv, low_threshold, high_threshold)

    # run Hough lines on edge image
    rho = rh  # distance resolution in pixels of the Hough grid
    theta = np.pi / tht  # angular resolution in radians of the Hough grid
    threshold = threshld  # votes needed
    min_line_length = min_ll
    max_line_gap = max_lg

    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    
    # draw lines onto blank image
    line_image = np.copy(img) * 0
    long_lines = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            if np.sqrt((y2-y1)**2 + (x2-x1)**2) > true_min_ll:  # set minimum length for lines included
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),thickness)
                long_lines.append([x1,x2,y1,y2])
                
    line_image = cv2.cvtColor(line_image,cv2.COLOR_BGR2GRAY)
    line_image_s = cv2.addWeighted(img, 0.8, line_image, 1, 0) 
    
    return line_image, line_image_s, long_lines

def get_trp_coords(starfilename):
    data = dict()
    with open(starfilename) as f:
        # Walk through line by line
        name = ""
        labels = []
        data = []
        mode = ""
        # Get the data
        rr = f.read().splitlines()
        l = len(rr)
        for line in rr:
            if line[0:5] == "data_":
                # gets the table name
                name = line
            elif line[0:5] == "loop_":
                # gets into a loop thing and tells the program to expect just labels
                mode = "labels"
            elif line[0:4] == "_rln":
                if mode == "labels": # get normal labels here
                    params = line.split()
                    labels.append(params[0][4:])
                else:
                # labels also hava data just behind
                    params = line.split()
                    labels.append(params[0])
                    if len(data) == 0:
                        data.append([])
                    data[0].append(params[1])
                    # since data came, set the mode
                    mode = "data"
            elif line == "":
                # emtpy row, closes table if data was read before
                """
                if mode == "data":
                    self.makeTable(starfilename, name, labels, tuple(data))
                    # Unset all the vars
                    name = ""
                    labels = []
                    data = []
                    mode = ""
                """
            else:
                # mode has to be labels or data before
                if mode == "labels" or mode == "data":

                    d = line.split()
                    if len(d) != 0:
                        # If there is empty fields, they will be filled with NULL
                        if len(d) < len(labels):
                            for i in range(len(labels)-len(d)):
                                d.append("NULL")
                        data.append(d)
                        mode = "data"
        
        data = np.array(data)
        data = data.astype(np.float)
        data = data.astype(np.int)
        
        return labels, data

def get_mrc_image(mrc_location, eq=False):
    imfile = mrcfile.open(mrc_location)
    imdata = imfile.data
    image = np.array(imdata)
    if eq:
    	image = exposure.equalize_hist(image)
    image = np.interp(image, (image.min(), image.max()), (0,255))
    image = image.astype(np.uint8)
    return image

def grab_troponin(image, x_coord, y_coord, grab_rad):
    xmin = x_coord - grab_rad if x_coord - grab_rad >= 0 else 0
    xmax = x_coord + grab_rad if x_coord + grab_rad <= image.shape[1] else image.shape[1]
    ymin = y_coord - grab_rad if y_coord - grab_rad >= 0 else 0
    ymax = y_coord + grab_rad if y_coord + grab_rad <= image.shape[0] else image.shape[0]

    troponin = [i[xmin:xmax] for i in image[ymin:ymax]]
    return troponin