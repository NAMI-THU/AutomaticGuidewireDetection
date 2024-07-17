import os

import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import matplotlib.pyplot as plt
from tqdm import tqdm


def convert(_ground_truth, _img_shape):
    x = float(_ground_truth[0])
    y = float(_ground_truth[1])
    w = float(_ground_truth[2])
    h = float(_ground_truth[3])

    # numpy for images is always HxWxD
    img_w = _img_shape[1]
    img_h = _img_shape[0]

    new_data = ((x - w / 2) * img_w, (y - h / 2) * img_h, (x + w / 2) * img_w, (y + h / 2) * img_h)
    return new_data


def check_tip_in_box(_true_tip_pos, _prediction):
    if _prediction is None:
        return False

    if (_prediction[0] <= _true_tip_pos[0] <= _prediction[2]
            and _prediction[1] <= _true_tip_pos[1] <= _prediction[3]):
        return True
    return False


def area(_box):
    if _box is None:
        return 0

    w = _box[2] - _box[0]
    h = _box[3] - _box[1]
    return w * h


if __name__ == '__main__':
    model = YOLO("models/lab/weights/best.pt")
    dataset = "data/LAB/"
    subset = "test"
    output = "evaluation/lab"
    conf_threshold = 0.1
    save_prediction_images = False

    image_folder = os.path.join(dataset, "images", subset)
    label_folder = os.path.join(dataset, "labels", subset)

    os.makedirs(output, exist_ok=True)

    file_names = ["File"]
    highest_conf_save = ["Conf of max conf box"]
    conf_size = ["Area of max conf box"]
    tip_inside = ["Tip inside highest prediction"]
    tip_inside_any = ["Tip inside any prediction"]
    predicted_on_background = ["Prediction on empty image"]

    predictions = model(image_folder, stream=True, verbose=False, device="cuda:0")
    for prediction in tqdm(predictions, "Analyzing", total=len(os.listdir(image_folder))):
        highest_conf = 0
        highest_box = None
        image_file = prediction.path.split(os.sep)[-1]
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

        img = prediction.orig_img
        annotator = Annotator(img) if save_prediction_images else None

        boxes = prediction.boxes
        all_boxes = []
        for j in range(boxes.shape[0]):
            box = boxes.xyxy[j].cpu().numpy()
            conf = boxes.conf[j].cpu().numpy()
            if conf >= conf_threshold:
                all_boxes.append(box)
                if save_prediction_images:
                    annotator.box_label(box, label=f"Prediction ({str(np.round(conf, 2))})", color=(200, 100, 0))
                if conf >= highest_conf:
                    highest_conf = conf
                    highest_box = box

        is_tip_inside_highest_box = "nan"
        is_tip_inside_any_box = "nan"
        if ground_truth is not None:
            predicted_on_background.append("nan")

            if save_prediction_images:
                annotator.box_label(convert(ground_truth, img.shape), label="Truth", color=(0, 255, 0))
            # numpy for images is always HxWxD
            true_tip_pos = (ground_truth[0] * img.shape[1], ground_truth[1] * img.shape[0])
            is_tip_inside_highest_box = check_tip_in_box(true_tip_pos, highest_box)
            is_tip_inside_any_box = any([check_tip_in_box(true_tip_pos, box) for box in all_boxes])
        else:
            # If the image was empty, e.g. no annotation available, check if the model still predicted something
            if len(all_boxes) >= 0:
                predicted_on_background.append(True)
            else:
                predicted_on_background.append(False)

        if save_prediction_images:
            img_with_boxes = annotator.result()
            plt.imshow(img_with_boxes)
            plt.savefig(os.path.join(output, f"out_{image_file}"))

        file_names.append(image_file)
        highest_conf_save.append(highest_conf)
        conf_size.append(area(highest_box))
        tip_inside.append(is_tip_inside_highest_box)
        tip_inside_any.append(is_tip_inside_any_box)

    zipped_data = [p for p in zip(file_names,
                                  highest_conf_save,
                                  conf_size,
                                  tip_inside,
                                  tip_inside_any,
                                  predicted_on_background)]
    np.savetxt(os.path.join(output, 'evaluation.csv'), zipped_data, delimiter=',', fmt='%s')
