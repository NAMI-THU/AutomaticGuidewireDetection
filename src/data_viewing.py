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

# Define a custom sorting key function
def get_capture_number(filename):
    print(filename)
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
  x_norm = x_shifted / width
  y_norm = y_shifted / height
  h_norm = bbox_h / height
  w_norm = bbox_w / width
  return x_norm, y_norm, h_norm, w_norm


# Specify the directory path where your JSON files are located
directory_path = '/content/drive/MyDrive/ColabNotebooks/yolo-object-detection-custom/data/processed/data-aurora/pos6'
# directory_path = '/drive/MyDrive/ColabNotebooks/yolo-object-detection-custom/data/processed/data_aurora/'

# Initialize an empty list to store the JSON data from each file
json_data_list = []
file_list = os.listdir(directory_path)
# print(file_list)
# file_list = [os.path.join(directory_path, filename) for filename in os.listdir(directory_path)]
# file_list = sorted(filter(os.path.isfile, os.listdir(directory_path)), key=os.path.getmtime)
file_list = sorted(file_list, key=get_capture_number)

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
    print(image_path)
    img = plt.imread(image_path)

    # # Convert the image to floating-point format
    # image_float = img.astype(np.float32)
    # # Find the minimum and maximum pixel values
    # min_pixel_value = np.min(image_float)
    # max_pixel_value = np.max(image_float)

    # img_norm = (image_float - min_pixel_value) / (max_pixel_value - min_pixel_value)

    fig, ax = plt.subplots()
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    # # Display the image
    ax.imshow(img)

    # Plot the red point
    ax.scatter(x_shifted, y_shifted, color='red', s=20)  # 's' controls the point size

    # Draw bounding box, values from YOLO version
    x_norm, y_norm, h_norm, w_norm = normalize_coords(x_shifted, y_shifted, img, 100, 100)
    # print(x_norm)
    # print(y_norm)
    # x_min, y_min, x_max, y_max = calculate_bounding_box(x_norm, y_norm, w_norm, h_norm)
    # print("edges ---------- ")
    # print(x_norm)
    # print(y_norm)
    # print(h_norm)
    # print(w_norm)
    # print("---------------- ")
    # bbox = patches.Rectangle((x_norm, y_norm), w_norm, h_norm, linewidth=2, edgecolor='r', facecolor='none')
    w_bbox = 20
    h_bbox = 20
    x_bot_left = x_shifted - (w_bbox / 2)
    y_bot_left = y_shifted - (h_bbox / 2)

    ax.add_patch(Rectangle((x_bot_left, y_bot_left), w_bbox, h_bbox, edgecolor='r', facecolor='none'))
    plt.imshow(np.flipud(img), cmap='gray', origin='lower')
    plt.show()
    break;
    # Saving annotation
    # class = 0
    # formatted_values = f"{class} {x_norm} {y_norm} {w_norm} {h_norm}"
    # with open("output.txt", "w") as file:
    #     file.write(formatted_values)
