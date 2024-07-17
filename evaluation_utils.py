def convert(_ground_truth, _img_shape=(1,1)):
    x = float(_ground_truth[0])
    y = float(_ground_truth[1])
    w = float(_ground_truth[2])
    h = float(_ground_truth[3])

    # numpy for images is always HxWxD
    img_w = _img_shape[1]
    img_h = _img_shape[0]

    new_data = ((x - w / 2) * img_w, (y - h / 2) * img_h, (x + w / 2) * img_w, (y + h / 2) * img_h)
    return new_data


def check_tip_in_box(_true_tip_pos, _prediction, allowed_threshold=0.):
    if _prediction is None:
        return False

    if (_prediction[0]-allowed_threshold <= _true_tip_pos[0] <= _prediction[2]+allowed_threshold
            and _prediction[1]-allowed_threshold <= _true_tip_pos[1] <= _prediction[3]+allowed_threshold):
        return True
    return False


def area(_box):
    if _box is None:
        return 0

    w = _box[2] - _box[0]
    h = _box[3] - _box[1]
    return w * h
