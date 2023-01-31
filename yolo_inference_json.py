import torch
import os
import glob
import cv2
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.hub.load('ultralytics/yolov5', 'custom', path="./runs/train/exp3/weights/best.pt")  # yolov5s
model.conf = 0.5  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.to(device)

image_dir = "./animals.v2-release.coco/test/images/"
image_path = glob.glob(os.path.join(image_dir, "*.jpg"))

img_info_path = "./animals.v2-release.coco/test/_annotations.coco.json"
with open(img_info_path, 'r', encoding='utf-8') as f:
    image_info = json.loads(f.read())

label_dict = {
    0: 'animal',
    1: 'cat',
    2: 'chicken',
    3: 'cow',
    4: 'dog',
    5: 'fox',
    6: 'goat',
    7: 'horse',
    8: 'person',
    9: 'racoon',
    10: 'skunk'
}

submission_anno = list()

for number, img_info in enumerate(image_info['images']):
    file_name = img_info['file_name']
    img_height = img_info['height']
    img_width = img_info['width']
    img_path = image_dir + file_name

    img = cv2.imread(img_path)
    result = model(img, size=640)
    # print(result)
    bbox = result.xyxy[0]
    image_name = os.path.basename(img_path)

    h, w, c = img.shape

    category_id = number + 1
    # print(image_name)
    # print(result)
    # print(result.xyxy)
    # print(bbox)
    #          x1 (pixels)  y1 (pixels)  x2 (pixels)  y2 (pixels)   confidence        class
    # tensor([[7.47613e+02, 4.01168e+01, 1.14978e+03, 7.12016e+02, 8.71210e-01, 0.00000e+00],
    #         [1.17464e+02, 1.96875e+02, 1.00145e+03, 7.11802e+02, 8.08795e-01, 0.00000e+00],
    #         [4.23969e+02, 4.30401e+02, 5.16833e+02, 7.20000e+02, 7.77376e-01, 2.70000e+01],
    #         [9.81310e+02, 3.10712e+02, 1.03111e+03, 4.19273e+02, 2.86850e-01, 2.70000e+01]])
    tmp_dict = dict()
    for box in bbox:
        x1 = box[0].item()
        y1 = box[1].item()
        x2 = box[2].item()
        y2 = box[3].item()
        x = int(x1)
        y = int(y1)
        w = int(x2)
        h = int(y2)
        cv2.rectangle(img, (x, y, w, h), (0, 255, 0), 2)
        print(w, h)

        class_number = box[5].item()
        class_number_int = int(class_number)
        labels = label_dict[class_number_int]

        sc = box[4].item()

        tmp_dict['bbox'] = [x, y, w, h]
        tmp_dict['category_id'] = category_id
        tmp_dict['area'] = w * h
        tmp_dict['image_id'] = img_info['id']
        tmp_dict['score'] = sc

        submission_anno.append(tmp_dict)

    cv2.imshow('test', img)
    cv2.waitKey(0)

with open('./animal.json', 'w', encoding='utf-8') as f:
    json.dump(submission_anno, f, indent=4, sort_keys=True, ensure_ascii=False)
