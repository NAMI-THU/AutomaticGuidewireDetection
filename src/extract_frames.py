import os
import numpy as np
import math
import random
import cv2




def convert_array_to_text_structure(arr,path):
    p = []
    for fn in arr:
        filename = path + '/' + fn
        # print(filename)
        p.append(filename + '\n')
    return p

def write_to_text(filename,data):
  
    # Creating file test.txt and writing 15% of lines in it
    with open(filename, 'w') as txt:
            # Going through all elements of the list
        for e in data:
                # Writing current path at the end of the file
            txt.write(e)

    return filename

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")
    return folder_path

def extract_frames(video_file_path, target_path, basename):
    # Defining VideoCapture object
    # video = cv2.VideoCapture(video_file_path)
    video = cv2.VideoCapture(video_file_path)

    # Check if the video file was successfully opened
    if not video.isOpened():
        print("Error: Could not open video.")
        return

    # Define writer that will write processed frames
    writer = None
    # Variables for spatial dimensions of the frames
    h, w = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    # variable for frame count
    f = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # variable for time
    t = 0

    # loop for catching frames

    for i in range(f):
        # capturing frame-by-frame
        ret, frame = video.read()
        # if frame not retrieved 
        # at the end of video
        # break the loop
        if not ret:
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #frame filename
        frame_filename = os.path.join(target_path, basename+"_"+str(i)+".jpg")

        # Save the frame as an image
        cv2.imwrite(frame_filename, gray_frame)


    # release video capture object and close any OpenCV windows
    video.release()
    cv2.destroyAllWindows()

# os.chdir(os.path.abspath(os.sep))



src_path = os.path.join(os.getcwd(),"data/raw/fluoroscopic_images")
target_path = os.path.join(os.getcwd(),"data/raw/")
os.chdir(src_path)
# src_texts = [os.path.join(src_path,filename) for filename in os.listdir() if filename.endswith('.txt')]     
# src_texts = ["cycle_1_fluo_val.txt","cycle_1_fluo_train.txt"] 
src_texts = ["cycle_1_fluo_test.txt"] 

# iterate through src_text
for fn in src_texts:
    # create a folder if train, text or validation
  
    # print(os.path.splitext(os.path.basename(fn))[0])
    folder_name = create_folder(os.path.join(target_path, os.path.splitext(os.path.basename(fn))[0]))
    # read text file
    try:
        # Open the file in read mode
        with open(fn, 'r') as file:
            print(fn,"File content line by line:")
            for line in file:
                print(line.strip())  # Print each line (stripping newline characters)
                # print("file exists?", os.path.exists(line))
                # print("---")
                # print(line.strip())
                # print("file exists?", os.path.exists(line.strip()))
                # print("---")
                fn_base = os.path.splitext(os.path.basename(line.strip()))[0]
                extract_frames(line.strip(), folder_name, fn_base)
                # break
    except FileNotFoundError:
        print("File not found.")

# read each line in the text file
# each line corresponds to 1 video
# convert video to frames
# save to associated folder with filename MVI_6973_x







  