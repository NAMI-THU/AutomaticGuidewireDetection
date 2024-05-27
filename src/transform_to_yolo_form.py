"""
FOR INDIVIDUAL DIRECTORY
"""

# [class centerx centery objwidth objheight] - Start with one folder first

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
import albumentations as A
# from PIL import Image

# Simple augmentation pipeline
transform = A.Compose([
    A.ToGray(p=1.0)
])
# visualize(augmented_image)

# Define a custom sorting key function
def get_capture_number(filename):
    # Split the filename by "_" and get the last part as a string
    capture_part = filename.split("_")[-1]
    # Remove any file extensions (e.g., ".png" or ".json") and convert to an integer
    return int(capture_part.split(".")[0])

# Calculate bounding box - Defining region for the tip
def calculate_bounding_box(center_x, center_y, width, height):
  x_min = center_x - (width / 2)
  y_min = center_y - (height / 2)
  x_max = center_x - (width / 2)
  y_max = center_y - (height / 2)
  return x_min, y_min, x_max, y_max

# normalized bounding box center
def normalize_coords(x_shifted, y_shifted, img, bbox_h, bbox_w):
  height, width, channels = img.shape
  # print(height)
  # print(width)
  x_norm = x_shifted / width
  y_norm = y_shifted / height
  h_norm = bbox_h / height
  w_norm = bbox_w / width
  return x_norm, y_norm, h_norm, w_norm


# Specify the directory path where your JSON files are located
directory_path = '/content/drive/MyDrive/ColabNotebooks/yolo-object-detection-custom/data/processed/data-aurora/pos6'
# directory_path_target = '/drive/MyDrive/Colab Notebooks/yolo-object-detection-custom/data/labelled-data'
directory_path_target = '/content/drive/MyDrive/ColabNotebooks/yolo-object-detection-custom/data/labelled/labelled_data/'


# Initialize an empty list to store the JSON data from each file
json_data_list = []
file_list = os.listdir(directory_path)
file_list = sorted(file_list, key=get_capture_number)

# # for testing only -------------
# # Numbers to match
# # numbers_to_match = [163, 182, 177, 165, 168, 170, 176, 173, 160, 166, 167]
# # numbers_to_match = [183,162,174,180,161,175,184,181]
# numbers_to_match = [164,171,169]

# # Create a regular expression pattern
# pattern = r'.*(' + '|'.join(str(num) for num in numbers_to_match) + r').*'

# # Filter the file list using the regular expression pattern
# filtered_files = [filename for filename in file_list if re.match(pattern, filename)]

# file_list = sorted(filtered_files, key=get_capture_number)

# # ------------------------------
src_path = os.path.join(os.getcwd(),"data/raw/guidewire")
target_path = os.path.join(os.getcwd(),"data/raw/")
os.chdir(src_path)
# src_texts = [os.path.join(src_path,filename) for filename in os.listdir() if filename.endswith('.txt')]     
src_texts = ["cycle_1_val2.txt","cycle_1_train.txt"]  

# Iterate over the files in the directory
for filename in file_list:
    basename, ext = os.path.splitext(filename)

    file_path = os.path.join(directory_path, basename+".json")
    with open(file_path, 'r') as json_file:
      json_data = json.load(json_file)
      val_toolName = json_data[0]["toolName"]
      x_shifted = json_data[0]["x_shifted"]
      y_shifted = json_data[0]["y_shifted"]
      z_shifted = json_data[0]["z_shifted"]

      x_raw = json_data[0]["x_raw"]
      y_raw = json_data[0]["y_raw"]
      z_raw = json_data[0]["z_raw"]

    image_path = os.path.join(directory_path, basename+".png")
    # print(image_path)
    img = plt.imread(image_path)
    fig, ax = plt.subplots()

    # augment image to gray scale
    augmented_image = transform(image=img)['image']
    # augmented_image = img
    ax.imshow(augmented_image)

    # Plot the red point
    ax.scatter(x_shifted, y_shifted, color='red', s=20)  # 's' controls the point size

    # Draw bounding box, values from YOLO version
    x_norm, y_norm, h_norm, w_norm = normalize_coords(x_shifted, y_shifted, img, 20, 20)
    # test
    # Check if x_norm and y_norm are within [0, 1]
    is_x_norm_valid = 0 <= x_norm <= 1
    is_y_norm_valid = 0 <= y_norm <= 1
    message = filename
    if not is_x_norm_valid:
        message += " invalid x"
    if not is_y_norm_valid:
        if not is_x_norm_valid:  # Append space if x is also invalid
            message += " "
        else:  # Append "invalid y" directly if x is valid
            message += "invalid y"

    # Print the message
    if not is_x_norm_valid or not is_y_norm_valid:
      print(message)

    # filename invalid x invalid y


    # print("-----")
    # print(x_norm)
    # print(y_norm)
    # print(h_norm)
    # print(w_norm)

    w_bbox = 20
    h_bbox = 20
    # x_bot_left = x_shifted - (w_bbox / 2) # x_min
    # y_bot_left = y_shifted - (h_bbox / 2) # y_max
    x_bot_left = x_shifted - (w_bbox / 2) # x_min
    y_bot_left = y_shifted - (h_bbox / 2) # y_max
    ax.add_patch(Rectangle((x_bot_left, y_bot_left), w_bbox, h_bbox, edgecolor='r', facecolor='none'))

    plt.imshow(np.flipud(augmented_image), cmap='gray', origin='lower')


    # change the file name
    output_filename = os.path.basename(directory_path)+"_"+basename

    # save bounding box in a text file
    cl_tip = 0
    # formatted_values = f"{cl_tip} {round(x_norm,4)} {round(y_norm,4)} {round(w_norm,4)} {round(h_norm,4)}"
    formatted_values = f"{cl_tip} {x_norm} {y_norm} {w_norm} {h_norm}"

    if not os.path.exists(os.path.join(directory_path_target, output_filename+".txt")):
      with open(os.path.join(directory_path_target, output_filename+".txt"), "x") as file:
          file.write(formatted_values)

    # save new grayed image to labelled-data
    plt.imsave(os.path.join(directory_path_target, output_filename+".png"),augmented_image, origin='lower')
    plt.show()
    # break;


