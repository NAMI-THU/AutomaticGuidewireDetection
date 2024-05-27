"""
This code transforms image to gray
and transforms IGT annotation to yolo form

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
# import albumentations as A
# from PIL import Image

# Simple augmentation pipeline
# transform = A.Compose([
#     A.ToGray(p=1.0)
# ])
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
def normalize_coords(x_shifted, y_shifted, img, bbox_h, bbox_w): # width, height
  height, width, channels = img.shape
  # print(height)
  # print(width)
  x_norm = x_shifted / width
  y_norm = y_shifted / height
  h_norm = bbox_h / height
  w_norm = bbox_w / width
  return x_norm, y_norm, h_norm, w_norm

def is_excluded(norm_vals_list):
      is_in_range = all(0 <= value <= 1 for value in norm_vals_list)
      is_excluded = False
      if is_in_range:
          is_excluded = False
      else:
          is_excluded = True
      return False if is_in_range == True else True

def created_folder(target_path,basename):
  path_yolo_form = os.path.join(target_path,basename.split("_")[-1])
  if not os.path.exists(path_yolo_form):
      os.makedirs(path_yolo_form)
      print(f"Folder '{path_yolo_form}' created.")
  else:
      print(f"Folder '{path_yolo_form}' already exists.")
  return path_yolo_form

def get_class(fname):
   tip_cl = {
      'GW': [0],
      'CT': [1],
      'CTGW' : [0,1]
   }
   cl = fname.split("_")[0]
   tip = 0
   if cl in tip_cl:
      tip = tip_cl[cl]
   else:
      tip = 4
   return tip

def get_tip_type(basename):
   cl = basename.split("_")[0]
   p = []
   if cl == "GW":
      p.append("gw_tip")
   if cl == "CT":
      p.append("ct_tip")
   if cl == "CTGW":
      p.append("ct_tip")
      p.append("gw_tip")
   return p

def convert_array_to_text_structure(class_tip,norms):
    p = []
    print("class_tip: ---")
    print(class_tip)
    for tip in class_tip:
        formatted_values = f"{tip} {norms[0]} {norms[1]} {norms[2]} {norms[3]}"
        # print(filename)
        p.append(formatted_values + '\n')
    return p

def write_to_text(filename,data):
  
    # Creating file test.txt and writing 15% of lines in it
    with open(filename, 'w') as txt:
            # Going through all elements of the list
        for e in data:
                # Writing current path at the end of the file
            txt.write(e)

    return filename
   

# Specify the directory path where your JSON files are located
directory_path = '/content/drive/MyDrive/ColabNotebooks/yolo-object-detection-custom/data/processed/data-aurora/pos6'
# directory_path_target = '/drive/MyDrive/Colab Notebooks/yolo-object-detection-custom/data/labelled-data'
directory_path_target = '/content/drive/MyDrive/ColabNotebooks/yolo-object-detection-custom/data/labelled/labelled_data/'



# Initialize an empty list to store the JSON data from each file
json_data_list = []
file_list = [filename for filename in os.listdir(directory_path) if not filename.endswith('.txt') and not filename.endswith('.csv')]
# print(file_list)
# file_list = sorted(file_list, key=get_capture_number)


# # ------------------------------
src_path = os.path.join(os.getcwd(),"data/custom_raw/custom_fluoroscopy/1_DATA_CLINIC/")
target_path = os.path.join(os.getcwd(),"data/custom_raw/custom_fluoroscopy/3_yolo_form_corrected_bbox/")

# os.chdir(src_path)
# src_texts = [os.path.join(src_path,filename) for filename in os.listdir() if filename.endswith('.txt')]     
src_texts = ["cycle.txt"]  
# src_texts = ["cycle_1_fluo_test.txt"]  

bbox_w = 15
bbox_h = 30
# Iterate over the files in the directory
for filename in src_texts:
    basename, ext = os.path.splitext(filename)
    print(basename)
    file_path = os.path.join(directory_path, basename+".txt")
    print(file_path)
    # print(file_path)
    with open(file_path, 'r') as txt_file:
      for line in txt_file:
        line = line.strip()
        print(line)
        
        header = os.path.basename(line)
        print(header)
        
        # print(cl_pos_ext)
        dir_files = set([os.path.splitext(filename)[0] for filename in os.listdir(line) if filename.endswith('.png') or filename.endswith('.json')])
        dir_files = sorted(dir_files, key=get_capture_number)
        # dir_files = set(dir_files)
        for item in dir_files:
          new_basename = header +"_"+ item.split("_")[1]
          new_name_img = new_basename+".png"
          new_name_anno = new_basename+".txt"
          # print(new_basename)

          og_img = os.path.join(line,item+".png")
          og_anno = os.path.join(line,item+".json")

          img = plt.imread(og_img)
          plt.imshow(np.flipud(img), cmap='gray', origin='lower')

          with open(os.path.join(line,item+".json"), 'r') as json_file:
            json_data = json.load(json_file)
           
            # print(json_data)
            val_toolName = json_data[0]["toolName"]
            x_shifted = json_data[0]["x_shifted"]
            y_shifted = json_data[0]["y_shifted"]
            z_shifted = json_data[0]["z_shifted"]

            x_raw = json_data[0]["x_raw"]
            y_raw = json_data[0]["y_raw"]
            z_raw = json_data[0]["z_raw"]

            
            x_norm, y_norm, h_norm, w_norm = normalize_coords(x_shifted, y_shifted, img, bbox_w, bbox_h) # width, height
            
            # First phase of data cleaning
            print(is_excluded([x_norm, y_norm, h_norm, w_norm]))

            # cl_tip = 0
            # formatted_values = f"{cl_tip} {x_norm} {y_norm} {w_norm} {h_norm}"
           
            # save image
            path_yolo_form = created_folder(target_path,basename)
            plt.imsave(os.path.join(path_yolo_form, new_name_img),img, origin='lower')
            
            cl_tip = [0]
            cl_text = write_to_text(os.path.join(path_yolo_form, "classes.txt"),get_tip_type(new_basename))
            norms = [x_norm, y_norm, h_norm, w_norm] 
            print("------- CL TIP")
            print(cl_tip)
            p_anno = convert_array_to_text_structure(cl_tip,norms)
           
            fn_anno = write_to_text(os.path.join(path_yolo_form, new_name_anno),p_anno)
            print(fn_anno)
          