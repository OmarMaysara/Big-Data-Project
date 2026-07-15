# Updated kitti_etl_clean.py with recommended enhancements

import os
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# -------------------------------------------------------------------------
# SPARK SESSION INITIALIZATION
# -------------------------------------------------------------------------

def initialize_spark_session(app_name="KittiETL"):
    """Creates and returns a configured SparkSession."""
    spark = SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    return spark

# --- Use this line at the start of your script ---
spark = initialize_spark_session()
print("PySpark Session Initialized!")
# -------------------------------------------------------------------------

ROOT = Path("E:/Road Object Detection Project")
RAW = ROOT / "raw"

IMAGE_DIR = RAW / "image" / "training" / "image_2"
LABEL_DIR = RAW / "label" / "training" / "label_2"
CALIB_DIR = RAW / "object_calib" / "training" / "calib"

CLEAN_DIR = ROOT / "clean"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# Typical real heights
CLASS_HEIGHTS = {
    "Car": 1.5,
    "Pedestrian": 1.7,
    "Cyclist": 1.7,
    "Van": 1.7,
    "Truck": 3.0
}

# ----------------------------------------------------
# PARSE CALIBRATION (now extracting fx, fy, cx, cy)
# ----------------------------------------------------
def parse_calib_file(path):
    with open(path) as f:
        lines = f.readlines()

    for L in lines:
        if L.startswith("P2:"):
            parts = L.strip().split()[1:]
            P2 = np.array([float(x) for x in parts]).reshape(3,4)
            fx = float(P2[0,0])
            fy = float(P2[1,1])
            cx = float(P2[0,2])
            cy = float(P2[1,2])
            return fx, fy, cx, cy
    return None, None, None, None


# ----------------------------------------------------
# PARSE LABEL_2 FORMAT
# ----------------------------------------------------
def parse_label_file(path):
    rows = []
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 15:
                continue

            cls = p[0]
            if cls not in ["Car", "Pedestrian", "Cyclist","Van","Truck"]:
                continue

            bbox_left = float(p[4])
            bbox_top = float(p[5])
            bbox_right = float(p[6])
            bbox_bottom = float(p[7])
            bbox_width = bbox_right - bbox_left
            bbox_height = bbox_bottom - bbox_top

            h = float(p[8])
            w = float(p[9])
            l = float(p[10])

            loc_x = float(p[11])
            loc_y = float(p[12])
            loc_z = float(p[13])  # forward depth

            alpha = float(p[14])

            rows.append({
                "class": cls,
                "bbox_left": bbox_left,
                "bbox_top": bbox_top,
                "bbox_right": bbox_right,
                "bbox_bottom": bbox_bottom,
                "bbox_width": bbox_width,
                "bbox_height": bbox_height,
                "3d_height": h,
                "3d_width": w,
                "3d_length": l,
                "loc_x": loc_x,
                "loc_y": loc_y,
                "loc_z": loc_z,
                "alpha": alpha
            })
    return rows


# ----------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------
records = []
image_files = sorted(list(IMAGE_DIR.glob("*.png")) + list(IMAGE_DIR.glob("*.jpg")))
print("Found images:", len(image_files))

for img_path in image_files:
    img_id = img_path.stem
    lbl_path = LABEL_DIR / f"{img_id}.txt"
    cal_path = CALIB_DIR / f"{img_id}.txt"

    if not lbl_path.exists() or not cal_path.exists():
        continue

    with Image.open(img_path) as im:
        img_w, img_h = im.size

    fx, fy, cx, cy = parse_calib_file(cal_path)
    objs = parse_label_file(lbl_path)

    if len(objs) == 0:
        records.append({
            "record_id": f"{img_id}_empty",
            "image_id": img_id,
            "file_name": img_path.name,
            "image_width": img_w,
            "image_height": img_h,
            "class": "None",
            "bbox_left": None,
            "bbox_top": None,
            "bbox_width": None,
            "bbox_height": None,
            "bbox_cx": None,
            "bbox_cy": None,
            "bbox_w_norm": None,
            "bbox_h_norm": None,
            "aspect_ratio": None,
            "bbox_area": 0,
            "true_depth_m": None,
            "true_distance_m": None,
            "fallback_distance_m": None,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy
    })
        continue


    for idx, o in enumerate(objs):
        cx_bbox = o["bbox_left"] + 0.5 * o["bbox_width"]
        cy_bbox = o["bbox_top"] + 0.5 * o["bbox_height"]

        bbox_w_norm = o["bbox_width"] / img_w
        bbox_h_norm = o["bbox_height"] / img_h

        aspect_ratio = o["bbox_width"] / max(o["bbox_height"], 1e-6)

        # True distances
        true_depth = o["loc_z"]
        euclid_dist = np.sqrt(o["loc_x"]**2 + o["loc_y"]**2 + o["loc_z"]**2)

        # Fallback distance
        if fx and o["bbox_height"] > 0:
            fallback_distance = (CLASS_HEIGHTS[o["class"]] * fx) / o["bbox_height"]
        else:
            fallback_distance = 1.0 / max(bbox_h_norm, 1e-6)

        record_id = f"{img_id}_{idx}"

        records.append({
            "record_id": record_id,
            "image_id": img_id,
            "file_name": img_path.name,
            "image_width": img_w,
            "image_height": img_h,
            "class": o["class"],
            "bbox_left": o["bbox_left"],
            "bbox_top": o["bbox_top"],
            "bbox_width": o["bbox_width"],
            "bbox_height": o["bbox_height"],
            "bbox_cx": cx_bbox,
            "bbox_cy": cy_bbox,
            "bbox_w_norm": bbox_w_norm,
            "bbox_h_norm": bbox_h_norm,
            "aspect_ratio": aspect_ratio,
            "bbox_area": o["bbox_width"] * o["bbox_height"],
            "true_depth_m": true_depth,
            "true_distance_m": euclid_dist,
            "fallback_distance_m": fallback_distance,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy
        })

# ----------------------------------------------------
# FINAL LOAD STEP (MODIFIED)
# ----------------------------------------------------

df_pandas = pd.DataFrame.from_records(records)
print("Total cleaned rows (Pandas):", len(df_pandas))

df_spark = spark.createDataFrame(df_pandas)
print("Spark DataFrame created successfully!")
df_spark.printSchema() # Inspect the inferred schema

clean_path = str(CLEAN_DIR / "kitti_clean_spark.parquet")
df_spark.coalesce(1).write.mode("overwrite").parquet(clean_path)

print(f"Saved cleaned dataset to Spark Parquet directory: {clean_path}")
spark.stop()
