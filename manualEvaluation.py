import os

import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import matplotlib.pyplot as plt
from tqdm import tqdm


def convert(groundtruth, imgshape):
    x = float(groundtruth[0])
    y = float(groundtruth[1])
    w = float(groundtruth[2])
    h = float(groundtruth[3])

    # numpy for images is always HxWxD
    img_w = imgshape[1]
    img_h = imgshape[0]

    new_data = ((x - w / 2) * img_w, (y - h / 2) * img_h, (x + w / 2) * img_w, (y + h / 2) * img_h)
    return new_data


def check_tip_in_box(true_tip_pos, prediction):
    if prediction is None:
        return False

    if (prediction[0] <= true_tip_pos[0] <= prediction[2]
            and prediction[1] <= true_tip_pos[1] <= prediction[3]):
        return True
    return False

def area(box):
    if box is None:
        return 0

    w = box[2] - box[0]
    h = box[3] - box[1]
    return w * h


if __name__ == '__main__':
    # TODO: Speed up by doing whole batch at once and setting to eval mode
    model = YOLO("models/lab/weights/best.pt")
    dataset = "data/LAB/"
    subset = "test"
    output = "evaluation/lab"
    conf_threshold = 0.1

    image_folder = os.path.join(dataset, "images", subset)
    label_folder = os.path.join(dataset, "labels", subset)

    os.makedirs(output, exist_ok=True)

    file_names = ["File"]
    highest_conf_save = ["Conf of max conf box"]
    conf_size = ["Area of max conf box"]
    tip_inside = ["Tip inside prediction"]

    collected_data = []
    ground_truths = []

    for image_file in tqdm(os.listdir(image_folder), desc="Preparing"):
        if "BG" in image_file:
            # TODO: Evaluate background too
            continue
        image_path = os.path.join(image_folder, image_file)
        label_path = os.path.join(label_folder, image_file.replace(".png", ".txt"))

        ground_truth = None
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                line = f.readline()
                bounding_box = line.split(" ")
                bounding_box = bounding_box[1:]
                to_float = []
                for v in bounding_box:
                    to_float.append(float(v))

                ground_truth = tuple(to_float)

        collected_data.append(image_path)
        ground_truths.append(ground_truth)

    predictions = model(collected_data, stream=True, verbose=False)
    for i, prediction in enumerate(tqdm(predictions,"Analysing")):
        image_file = collected_data[i].split("/")[-1]
        ground_truth = ground_truths[i]
        img = prediction[0].orig_img
        annotator = Annotator(img)
        boxes = prediction[0].boxes
        highest_conf = 0
        highest_box = None

        for i in range(boxes.shape[0]):
            box = boxes.xyxy[i].cpu().numpy()
            conf = boxes.conf[i].cpu().numpy()
            if conf >= conf_threshold:
                annotator.box_label(box, label=f"Prediction ({str(np.round(conf, 2))})", color=(200, 100, 0))
                if conf >= highest_conf:
                    highest_conf = conf
                    highest_box = box

        is_tip_inside = "not valid"
        if ground_truth is not None:
            annotator.box_label(convert(ground_truth, img.shape), label="Truth", color=(0, 255, 0))
            # numpy for images is always HxWxD
            true_tip_pos = (ground_truth[0]*img.shape[1], ground_truth[1]*img.shape[0])
            is_tip_inside = check_tip_in_box(true_tip_pos, highest_box)

        img_with_boxes = annotator.result()
        plt.imshow(img_with_boxes)
        plt.savefig(os.path.join(output, f"out_{image_file}"))

        file_names.append(image_file)
        highest_conf_save.append(highest_conf)
        conf_size.append(area(highest_box))
        tip_inside.append(is_tip_inside)

    np.savetxt('evaluation.csv', [p for p in zip(file_names, highest_conf_save,conf_size,tip_inside)], delimiter=',', fmt='%s')
