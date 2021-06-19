"""Batch resizing the training data

Source: https://www.kaggle.com/phunghieu/gwd-resize-images-bboxes
"""

import json
from pathlib import Path

import typer
import numpy as np
import pandas as pd
import cv2
import albumentations as A
from tqdm import tqdm


def load_dataframe(csv_path: Path, image_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Merge all bboxes of each corresponding image
    # Format: [[x1 y1 w1 h1], [x2 y2 w2 h2], [x3 y3 w3 h3], ...]
    df.bbox = df.bbox.apply(lambda x: " ".join(np.array(json.loads(x), dtype=str)))
    df.bbox = df.groupby(["image_id"]).bbox.transform(lambda x: "|".join(x))
    df.drop_duplicates(inplace=True, ignore_index=True)
    df.bbox = df.bbox.apply(
        lambda x: np.array(
            [item.split(" ") for item in x.split("|")], dtype=np.float32
        ).tolist()
    )

    # Create a path to each image
    df["image_path"] = df.image_id.apply(lambda x: str(image_dir / (x + ".jpg")))

    return df


def load_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return image


def fix_out_of_range(bbox: list, max_size: int = 1024) -> list:
    bbox[2] = min(bbox[2], max_size - bbox[0])
    bbox[3] = min(bbox[3], max_size - bbox[1])
    return bbox


def main(image_size: int, root: str = "data/"):
    root_path = Path(root)
    train_dir = root_path / "train"
    target_dir = root_path / f"{image_size}"
    (target_dir / "train").mkdir(parents=True)

    df = load_dataframe(root_path / "train.csv", train_dir)

    transform = A.Compose(
        [
            A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_AREA),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="coco", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )

    list_of_image_ids = []
    list_of_bboxes = []
    list_of_sources = []

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        image = load_image(row.image_path)
        bboxes = row.bbox

        # Fix "out-of-range" bboxes
        bboxes = [fix_out_of_range(bbox) for bbox in bboxes]

        result = transform(image=image, bboxes=bboxes, labels=np.ones(len(bboxes)))
        new_image = result["image"]
        new_bboxes = np.array(result["bboxes"]).tolist()

        # Save new image
        cv2.imwrite(str(target_dir / "train" / (row.image_id + ".jpg")), new_image)

        for new_bbox in new_bboxes:
            list_of_image_ids.append(row.image_id)
            list_of_bboxes.append(new_bbox)
            list_of_sources.append(row.source)

    new_data_dict = {
        "image_id": list_of_image_ids,
        "width": [image_size] * len(list_of_image_ids),
        "height": [image_size] * len(list_of_image_ids),
        "bbox": list_of_bboxes,
        "source": list_of_sources,
    }
    new_df = pd.DataFrame(new_data_dict)
    new_df.to_csv(target_dir / "train.csv", index=False)


if __name__ == "__main__":
    typer.run(main)
