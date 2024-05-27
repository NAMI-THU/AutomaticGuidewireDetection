# # $bbox_interval=1
# # $save_period=1
# epoch_clinic=1000 #1000
# epoch_lab=1000 #1000
epoch_newdata=500 #1000


# # 1. CLINICAL DATA
# echo "1. TRAIN ON CLINICAL DATASET"
# python yolov5//train.py --img 640 --batch 64 --epochs $epoch_clinic --data "../data/guidewiredetection_clinic.yaml" --weights yolov5s.pt --bbox_interval 1
# newest_folder_train=$(ls -lt "yolov5/runs/train/" | grep '^d' | head -n 1 | awk '{print $9}')
# echo "Trained weights: $newest_folder_train"

# echo "2. VALIDATION ON TEST DATASET"
# python yolov5/val.py --weights yolov5/runs/train/"$newest_folder_train"/weights/best.pt --data "../data/guidewiredetection_clinic.yaml" --img 640 --task test --conf-thres 0.5 --save-hybrid

# echo "3. VALIDATION ON VAL DATASET"
# python yolov5/val.py --weights yolov5/runs/train/"$newest_folder_train"/weights/best.pt --data "../data/guidewiredetection_clinic.yaml" --img 640 --conf-thres 0.5 --save-hybrid

# echo "4. DETECT ON TEST DATASET"
# python yolov5/detect.py --weights yolov5/runs/train/"$newest_folder_train"/weights/best.pt --source '../data/CLINIC_TEST_SR_100/images/D_test_sr'



# # 2. LAB DATA
# echo "1. TRAIN ON LAB DATASET"
# python yolov5/train.py --img 640 --batch 64 --epochs $epoch_lab --data "../data/guidewiredetection_lab.yaml" --weights yolov5s.pt --bbox_interval 1
# newest_folder_train_auto=$(ls -lt "yolov5/runs/train/" | grep '^d' | head -n 1 | awk '{print $9}')
# echo "Trained weights: $newest_folder_train_auto"

# echo "2. VALIDATION ON TEST DATASET"
# python yolov5/val.py --weights yolov5/runs/train/"$newest_folder_train_auto"/weights/best.pt --data "../data/guidewiredetection_lab.yaml" --img 640 --task test --conf-thres 0.5 --save-hybrid

# echo "3. VALIDATION ON VAL DATASET"
# python yolov5/val.py --weights yolov5/runs/train/"$newest_folder_train_auto"/weights/best.pt --data "../data/guidewiredetection_lab.yaml" --img 640 --conf-thres 0.5  --save-hybrid

# echo "4. DETECT ON TEST DATASET"
# python yolov5/detect.py --weights yolov5/runs/train/"$newest_folder_train_auto"/weights/best.pt --source '../data/LAB_TEST_SR_100/images/test/'

# 3. NEW DATA
echo "1. TRAIN ON NEW DATASET"
python yolov5/train.py --img 640 --batch 64 --epochs $epoch_newdata --data "../data/guidewiredetection_newdata.yaml" --weights yolov5s.pt --bbox_interval 1
newest_folder_train_auto=$(ls -lt "yolov5/runs/train/" | grep '^d' | head -n 1 | awk '{print $9}')
echo "Trained weights: $newest_folder_train_auto"

echo "2. VALIDATION ON TEST DATASET"
python yolov5/val.py --weights yolov5/runs/train/"$newest_folder_train_auto"/weights/best.pt --data "../data/guidewiredetection_newdata.yaml" --img 640 --task test --conf-thres 0.5 --save-hybrid

echo "3. VALIDATION ON VAL DATASET"
python yolov5/val.py --weights yolov5/runs/train/"$newest_folder_train_auto"/weights/best.pt --data "../data/guidewiredetection_newdata.yaml" --img 640 --conf-thres 0.5  --save-hybrid

# echo "4. DETECT ON TEST DATASET"
# python yolov5/detect.py --weights yolov5/runs/train/"$newest_folder_train_auto"/weights/best.pt --source '../data/LAB_TEST_SR_100/images/test/'