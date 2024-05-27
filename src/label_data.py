# With Dummy Images from G7x ii

import os
import re
import json
from PIL import Image, ImageDraw

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np

import matplotlib.patches as patches
import pandas as pd
from matplotlib.patches import Rectangle
import cv2


def write_to_text(filename,data):
  
    # Creating file test.txt and writing 15% of lines in it
    with open(filename, 'w') as txt:
            # Going through all elements of the list
        for e in data:
                # Writing current path at the end of the file
            txt.write(e)

    return filename

# Define a custom sorting key function
def get_capture_number(filename):
    # print(filename)
    # Split the filename by "_" and get the last part as a string
    capture_part = filename.split("_")[-1]
    print(filename.split("_")[1])
    # Remove any file extensions (e.g., ".png" or ".json") and convert to an integer
    return int(capture_part.split(".")[0])

def sorted_files(filenames):
    numerical_parts = [int(re.search(r'\d+', item).group()) for item in filenames]
    return sorted(zip(numerical_parts, filenames))

def custom_sort(tup):
    # Sort first by the first element, then by the second element
    return (tup[0], tup[1])

# Specify the directory path where your JSON files are located
src_path = os.path.join(os.getcwd(),"data/raw/")
target_path = os.path.join(os.getcwd(),"data/raw/")
os.chdir(src_path)
# src_texts = [os.path.join(src_path,filename) for filename in os.listdir() if filename.endswith('.txt')]     
src_directory = ["cycle_1_train","cycle_1_val","cycle_1_test"]  




for dir in src_directory:
   path_dir = os.path.join(src_path,dir)
   file_list = os.listdir(path_dir)
   filenames_unsorted = [(int(os.path.splitext(filename)[0].split('_')[1]),int(os.path.splitext(filename)[0].split('_')[2])) for filename in file_list if filename.endswith(".txt") and filename.find("classes") == -1 and filename.find("cycle") == -1]
  #  filenames = [t[1] for t in sorted_files(filenames)]
   sorted_filenames = sorted(filenames_unsorted, key=custom_sort)
   filenames = [ os.path.join(path_dir,"MVI_"+str(filename[0])+"_"+str(filename[1])) for filename in sorted_filenames]

  # Iterate over the files in the directory
   p = []
   text_file_to_write = os.path.join(path_dir,dir+".txt")
   for filename in filenames:
      #  print(filenames)
      # basename = os.path.splitext(os.path.basename(filename))[0]
      txt_file = filename+".txt"
      frame_file = filename+".jpg"
    
  #     # print("writing to this file...",text_file_to_write)
      p.append(frame_file + "\n")

  #  print(p)
  #  text_file = write_to_text(text_file_to_write,file_list)
       # Creating file test.txt and writing 15% of lines in it
   with open(text_file_to_write, 'w') as txt:
            # Going through all elements of the list
        for e in p:
                # Writing current path at the end of the file
            txt.write(e)


        # cv2.waitKey(0)
        # cv2.destroyAllWindows()



  # break

# for filename in file_list:
#     basename, ext = os.path.splitext(filename)

#     file_path = os.path.join(directory_path, basename+".txt")
#     with open(file_path, 'r') as file:
#       lines = file.readlines()
#       print("-------------")

#       for line in lines:
#         print(line)
#         # Remove leading and trailing whitespace and split by spaces
#         values = line.strip().split()
#         # xmin, ymin, width, height
#         print (values)

    #     # Now 'values' contains a list of values from the line
    #     # print(values)
    #     # cl_tip = int(values[0])
    #     # x_center = float(values[1])
    #     # y_center = float(values[1])
    #     # width = float(values[1])
    #     # height = float(values[1])
    #     x_shifted = json_data[0]["x_shifted"]
    #     y_shifted = json_data[0]["y_shifted"]
    #     z_shifted = json_data[0]["z_shifted"]

    #     x_raw = json_data[0]["x_raw"]
    #     y_raw = json_data[0]["y_raw"]
    #     z_raw = json_data[0]["z_raw"]

    # image_path = os.path.join(directory_path, basename+".png")
    # print(image_path)
    # img = plt.imread(image_path)

    # fig, ax = plt.subplots()
    # ax.imshow(img)

    # # Plot the red point
    # ax.scatter(x_shifted, y_shifted, color='red', s=20)  # 's' controls the point size

    # # Draw bounding box, values from YOLO version
    # x_norm, y_norm, h_norm, w_norm = normalize_coords(x_shifted, y_shifted, img, 100, 100)
    # w_bbox = 20
    # h_bbox = 20
    # x_bot_left = x_shifted - (w_bbox / 2)
    # y_bot_left = y_shifted - (h_bbox / 2)

    # ax.add_patch(Rectangle((x_bot_left, y_bot_left), w_bbox, h_bbox, edgecolor='r', facecolor='none'))
    # plt.imshow(np.flipud(img), cmap='gray', origin='lower')
    # plt.show()




