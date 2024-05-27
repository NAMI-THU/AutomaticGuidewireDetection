import os
import numpy as np
import math
import random

def convert_array_to_text_structure(arr,path):
    p = []
    for fn in arr:
        filename = path + '/' + fn
        # print(filename)
        p.append(filename + '\n')
    return p

def get_split_ratio(part):
    train_data = math.floor(part * train_data_per)
    validation_data = (part - train_data) / 2 if (part - train_data) % 2 == 0 else round((part - train_data) / 2)
    test_data = part - (train_data + validation_data)
    return train_data, validation_data,test_data

def write_to_text(filename,data):
  
    # Creating file test.txt and writing 15% of lines in it
    with open(filename, 'w') as txt:
            # Going through all elements of the list
        for e in data:
                # Writing current path at the end of the file
            txt.write(e)

    return filename



path = os.path.join(os.getcwd(),"data/custom_raw/custom_fluoroscopy/1_DATA_CLINIC")
target_path_splits = path
os.chdir(path)
# filenames = list(os.listdir())
filenames = [filename for filename in os.listdir()]
random.shuffle(filenames)
print(filenames)
data_size = len(filenames)
print(data_size)
filenames_shuffled = random.sample(filenames, data_size)


k = 1
train_data_per = 0.8
validation_data_per = 0.15
test_data_per = 0.15


# determines percentage for train, validation and test
interval_arr = np.linspace(0, data_size, k+1, dtype=int)
parts_list = [interval_arr[i+1] - interval_arr[i] for i in range(k)]
print(parts_list)
split_ratio_per_cycle = []

start = 0
for i,group_size_p_cycle in enumerate(parts_list):
    # diving data to k
    end = start + group_size_p_cycle
    cycle_data = filenames_shuffled[start:end]
    print("CYCLE_DATA----")
    print(cycle_data)
    start = end
    
    # getting train:validation:test ratio for every cycle 
    rat_train, rat_val, rat_test = get_split_ratio(group_size_p_cycle)
    print("Split Ratio")
    print(rat_train)
    print(rat_val)
    print(rat_test)
  

    # split filenames for training, validation and testing
    p_train_data = convert_array_to_text_structure(cycle_data[:rat_train],path)
    p_val_data = convert_array_to_text_structure(cycle_data[int(len(p_train_data)):int(len(p_train_data))+int(rat_val)],path)
    p_test_data = convert_array_to_text_structure(cycle_data[int(len(p_val_data)):int(len(p_val_data))+int(rat_test)],path)

    # write the list in the text file
    fn_train = write_to_text(os.path.join(target_path_splits,"cycle_"+str(i+1)+"_fluo_train.txt"),p_train_data)
    fn_val = write_to_text(os.path.join(target_path_splits,"cycle_"+str(i+1)+"_fluo_val.txt"),p_val_data)
    fn_test = write_to_text(os.path.join(target_path_splits,"cycle_"+str(i+1)+"_fluo_test.txt"),p_test_data)


    # print("p_train", p_train_data, len(p_train_data))
    # # p_val = cycle_data[int(len(p_train)):int(len(p_train))+rat_val]
    # print("p_val", p_val_data, len(p_val_data))
    # # p_test = cycle_data[int(len(p_val)):int(len(p_val))+rat_test]
    # print("p_test", p_test_data, len(p_test_data))

    # saving to text files
    



    # split_ratio_per_cycle.append([rat_train, rat_val, rat_test, cycle_data, len(cycle_data)])



# print(split_ratio_per_cycle)

# for i, info in enumerate(split_ratio_per_cycle):
#     print(info)
#     rat_train = info[0]
#     rat_train = info[0]
#     p_train = info




# data_group = []
# for i,group_size_p_cycle in enumerate(parts_list):
#     end = start + group_size_p_cycle
#     print(group_size_p_cycle)
#     cycle_data = filenames_shuffled[start:end]
#     start = end
#     data_group.append(cycle_data)
#     # print(i,":",cycle_data)

# print()


# split
# p = []
# random_dataset = random.sample(filenames, data_size)
# for cycle, data_ratio in enumerate(data_per_cycle):
#     print("cycle",cycle+1)
#     p_train = random_dataset[:int(len(p) * 0.15)]
#     # p_test = p[:int(len(p) * 0.15)]

    # print(p_test)

    # # Deleting from initial list first 15% of elements
    # p = p[int(len(p) * 0.15):]


# random_dataset = random.sample(filenames, data_size)
# print(random_dataset)