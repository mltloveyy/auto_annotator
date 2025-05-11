import json
import os
import shutil

import cv2


def labelme2yolo(points, imgsz):
    x_min = min([p[0] for p in points])
    x_max = max([p[0] for p in points])
    y_min = min([p[1] for p in points])
    y_max = max([p[1] for p in points])

    dw = 1.0 / imgsz[0]
    dh = 1.0 / imgsz[1]
    x = (x_min + x_max) / 2.0
    y = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def labelme2bbox(points):
    x_min = min([p[0] for p in points])
    x_max = max([p[0] for p in points])
    y_min = min([p[1] for p in points])
    y_max = max([p[1] for p in points])

    w = x_max - x_min
    h = y_max - y_min
    return (int(x_min), int(y_min), int(w), int(h))


def get_classes(root_dir, classes, check=False):
    if check:
        check_dir = root_dir + "_check"
        if os.path.exists(check_dir):
            shutil.rmtree(check_dir)
    for filename in os.listdir(root_dir):
        if filename.endswith(".json"):
            json_path = os.path.join(root_dir, filename)
            with open(json_path, "r") as f:
                data = json.load(f)

            if check:
                image_path = os.path.join(root_dir, data["imagePath"])
                image = cv2.imread(image_path)

            for i, shape in enumerate(data["shapes"]):
                category = shape["label"]
                if category not in classes:
                    classes.append(category)
                if check:
                    category_dir = os.path.join(check_dir, category)
                    if not os.path.exists(category_dir):
                        os.makedirs(category_dir)

                    x, y, w, h = labelme2bbox(shape["points"])
                    target = image[y : y + h, x : x + w]
                    target_path = os.path.join(category_dir, f"{os.path.splitext(filename)[0]}_{i}.png")

                    cv2.imwrite(target_path, target)


def json2txt(classes, root_dir, save_dir, ignores=None):
    for filename in os.listdir(root_dir):
        if filename.endswith(".json"):
            json_path = os.path.join(root_dir, filename)
            with open(json_path, "r") as f:
                data = json.load(f)
            if len(data["shapes"]) == 0:
                continue
            imgsz = (data["imageWidth"], data["imageHeight"])
            valid = False
            with open(os.path.join(save_dir, os.path.splitext(filename)[0] + ".txt"), "w") as out_file:
                for shape in data["shapes"]:
                    label = shape["label"]
                    if label in ignores:
                        if label in classes:
                            classes.remove(label)
                        continue
                    xywh = labelme2yolo(shape["points"], imgsz)
                    out_file.write(f"{classes.index(label)} {' '.join(map(str, xywh))}\n")
                    valid = True
            if valid and save_dir != root_dir:
                shutil.copy(os.path.join(root_dir, data["imagePath"]), os.path.join(save_dir, data["imagePath"]))


if __name__ == "__main__":
    root_dirs = [
        "/home/yy/workspace/datasets/railwaytools/annotation/20241115",
        "/home/yy/workspace/datasets/railwaytools/annotation/20241207",
    ]
    save_dir = "/home/yy/workspace/datasets/railwaytools/train"

    ignores = [
        "yaoshi",
        "saichi",
        "chongzi",
        "taotongjiegan",
        "zhijiaotaotongluosidao",
        "zhediedao",
        "gangchi",
        "jiandao",
    ]

    classes = []
    for dir in root_dirs:
        get_classes(dir, classes, check=False)

    for dir in root_dirs:
        json2txt(classes, dir, save_dir, ignores)

    for i, c in enumerate(classes):
        print(f"  {i}: {c}")
