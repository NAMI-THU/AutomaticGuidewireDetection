from ultralytics import YOLO
from PIL import Image

if __name__ == '__main__':
    model = YOLO("yolov8l.pt")
    results = model.train(data="data/guidewiredetection_clinic.yaml", epochs=300, device=[0, 1])

    # for i, r in enumerate(results):
    #     # Plot results image
    #     im_bgr = r.plot()  # BGR-order numpy array
    #     im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
    #
    #     # Show results to screen (in supported environments)
    #     r.show()
    #
    #     # Save results to disk
    #     r.save(filename=f"results{i}.jpg")