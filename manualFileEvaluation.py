import os

import numpy as np
from tqdm import tqdm

from evaluation_utils import check_tip_in_box, convert


def read_prediction(directory, file):
    file_path = os.path.join(directory, file)
    if os.path.exists(file_path):
        boxes = []
        with open(file_path, "r") as f:
            line = f.readline()
            while line is not None and not line == "":
                bounding_box = line.split(" ")
                bounding_box = bounding_box[1:-1]
                confidence = bounding_box[-1]
                to_float = []
                for v in bounding_box:
                    to_float.append(float(v))

                box = {"box":tuple(to_float),"conf":float(confidence)}
                boxes.append(box)
                line = f.readline()
        boxes = sorted(boxes, key=lambda k: k['conf'], reverse=True)
        return boxes
    else:
        return None

def analyze_from_file(predictions_path: str, dataset: str, subdata: str, output_path: str):
    allowed_threshold = 0.01
    image_folder = os.path.join(dataset, "images", subdata)
    label_folder = os.path.join(dataset, "labels", subdata)

    os.makedirs(output_path, exist_ok=True)

    file_names = ["File"]
    tip_inside = ["Tip inside highest prediction"]
    tip_inside_any = ["Tip inside any prediction"]
    predicted_on_background = ["Prediction on empty image"]
    contains_annotation = ["Contains tip"]

    files = os.listdir(image_folder)

    for file in tqdm(files, f"Analyzing {image_folder}", total=len(files)):
        if file == "classes.txt":
            continue
        label_path = os.path.join(label_folder, file.replace(".png", ".txt"))
        prediction_boxes = read_prediction(predictions_path, file.replace(".png", ".txt"))

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

        is_tip_inside_highest_box = "nan"
        is_tip_inside_any_box = "nan"
        if ground_truth is not None:
            predicted_on_background.append("nan")
            contains_annotation.append(True)

            true_tip_pos = (ground_truth[0], ground_truth[1])
            if prediction_boxes is None:
                is_tip_inside_highest_box = False
            else:
                is_tip_inside_highest_box = check_tip_in_box(true_tip_pos, convert(prediction_boxes[0]["box"]),allowed_threshold)
                is_tip_inside_any_box = any([check_tip_in_box(true_tip_pos, convert(box["box"]),allowed_threshold) for box in prediction_boxes])
        else:
            contains_annotation.append(False)
            # If the image was empty, e.g. no annotation available, check if the model still predicted something
            if prediction_boxes is not None and len(prediction_boxes) > 0:
                predicted_on_background.append(True)
            else:
                predicted_on_background.append(False)

        file_names.append(file)
        tip_inside.append(is_tip_inside_highest_box)
        tip_inside_any.append(is_tip_inside_any_box)

    zipped_data = [p for p in zip(file_names,
                                  contains_annotation,
                                  tip_inside,
                                  tip_inside_any,
                                  predicted_on_background)]
    np.savetxt(os.path.join(output_path, 'evaluation.csv'), zipped_data, delimiter=',', fmt='%s')

    files_with_annotations = [x for x in range(len(contains_annotation)) if contains_annotation[x] is True]
    correct_tips = sum([1. for x in files_with_annotations if tip_inside[x] is True])
    # Interpretation for this case:
    # TP: Tip exists and was correctly detected
    true_positives = correct_tips/len(files_with_annotations)

    predicted_on_background_data_length = len([1. for x in range(len(predicted_on_background)) if predicted_on_background[x] is True or predicted_on_background[x] is False])
    predicted_on_background_true = sum([1. for x in range(len(predicted_on_background)) if predicted_on_background[x] is True])
    predicted_on_background_false = sum([1. for x in range(len(predicted_on_background)) if predicted_on_background[x] is False])

    # FP: No tip exists, but a prediction was done
    if predicted_on_background_true == 0:
        false_positives = 0.
    else:
        false_positives = predicted_on_background_true/predicted_on_background_data_length

    # TN: No tip exists and also no prediction was done
    if predicted_on_background_false == 0:
        true_negatives = 0.
    else:
        true_negatives = predicted_on_background_false/predicted_on_background_data_length

    # FN: A tip exists, but was not correctly detected (but possible something was predicted somewhere else)
    false_negatives = sum([1. for x in files_with_annotations if tip_inside[x] is False])/len(files_with_annotations)
    # Accuracy: When a tip was there, how often was it recognized?
    return true_positives, false_positives, true_negatives, false_negatives


if __name__ == '__main__':
    for task in ["clinic","lab"]:
        data = f"data/{task.upper()}/"
        subset = "test"
        output = f"evaluation-yolo5/{task}"
        label_path = f"data/yolo5annotations/{task}"
        tp, fp, tn, fn = analyze_from_file(label_path, data, subset, output)
        print(f"Results for {task}:\n"
              f"\tTip existed and was correctly detected:\t\t\t{round(tp*100,2)}%\n"
              f"\tNo tip exists, but a prediction was done:\t\t{round(fp*100,2)}%\n"
              f"\tNo tip exists and no prediction was done:\t\t{round(tn*100,2)}%\n"
              f"\tA tip exists, but was not *correctly* detected\t{round(fn*100,2)}%")
