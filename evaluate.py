from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("models/train11-big-2epochs/weights/best.pt")
    # result = model("data/CLINIC_TEST_SR_100/images/D_test_sr/SR_p2_og_hf_vf_55.png")
    # print(result)
    # result[0].show()
    metrics = model.val(data="data/guidewiredetection_clinic.yaml", split="test", save_json=True)
    print(metrics.box.map)  # mAP50-95
    print(metrics.box.map50)  # mAP50
    print(metrics.box.map75)  # mAP75
    print(metrics.box.maps)  # list of mAP50-95 for each category