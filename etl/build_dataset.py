import pandas as pd
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# CONFIGURATION (ABSOLUTE PATHS - GUARANTEED)
# ---------------------------------------------------------

# Path to the single, clean Parquet file produced by Spark.
# We are using Path(r"...") for guaranteed interpretation of the absolute path.
PARQUET_FILE = Path(r"E:\Road Object Detection Project\clean\kitti_clean_spark.parquet\part-00000-4856f202-cfc2-4668-b7d4-c3acb5191507-c000.snappy.parquet")

# Source directory containing the raw images.
SOURCE_IMAGES_DIR = Path(r"E:\Road Object Detection Project\raw\image\testing\image_2")

# Root directory where the final YOLO-formatted dataset will be built.
DATASET_ROOT = Path(r"E:\Road Object Detection Project\organized\dataset")

# Class mapping
CLASS_MAP = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}

# ---------------------------------------------------------
# 1. LOAD AND PREPARE DATA
# ---------------------------------------------------------
print(f"Loading data from: {PARQUET_FILE}")

# We explicitly convert the Path object to a string for pd.read_parquet
try:
    df = pd.read_parquet(str(PARQUET_FILE))
except FileNotFoundError:
    print(f"FATAL ERROR: Parquet file not found. Checked path: {PARQUET_FILE.absolute()}")
    print("Please ensure the file exists at this exact location.")
    exit()
except Exception as e:
    print(f"An error occurred while reading the Parquet file: {e}")
    exit()

# Filter only the classes we care about
df = df[df["class"].isin(CLASS_MAP.keys())].copy()

# Map class names to numeric IDs
df["class_id"] = df["class"].map(CLASS_MAP)

print(f"Loaded {len(df)} labels for {len(CLASS_MAP)} classes.")

# ---------------------------------------------------------
# 2. SPLIT DATA
# ---------------------------------------------------------
# Split based on unique image filenames to avoid data leakage
unique_files = df["file_name"].unique()
train_files, val_files = train_test_split(unique_files, test_size=0.2, random_state=42)

# Create dataframes for split based on file names
train_df = df[df["file_name"].isin(train_files)]
val_df = df[df["file_name"].isin(val_files)]

print(f"Training Images: {len(train_files)} | Validation Images: {len(val_files)}")

# ---------------------------------------------------------
# 3. CREATE DIRECTORY STRUCTURE
# ---------------------------------------------------------
# This creates the necessary train/val subdirectories for images and labels.
for split in ["train", "val"]:
    (DATASET_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
    (DATASET_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 4. PROCESSING FUNCTION
# ---------------------------------------------------------
def process_split(split_df, split_name):
    print(f"Processing {split_name} split...")
    
    # Group by filename so we write one label file per image
    grouped = split_df.groupby("file_name")
    
    count = 0
    total_to_process = len(grouped)
    
    for img_name, group in grouped:
        # 1. Copy Image
        src_img_path = SOURCE_IMAGES_DIR / img_name
        dst_img_path = DATASET_ROOT / "images" / split_name / img_name
        
        if not src_img_path.exists():
            # 
            print(f"Warning: Image {img_name} not found at {src_img_path}. Skipping corresponding labels.")
            continue
            
        shutil.copy(src_img_path, dst_img_path)
        
        # 2. Write Label File (YOLO format: class x_center y_center width height)
        label_filename = Path(img_name).with_suffix(".txt").name
        dst_label_path = DATASET_ROOT / "labels" / split_name / label_filename
        
        with open(dst_label_path, "w") as f:
            for _, row in group.iterrows():
                # Normalized Center Coordinates (X, Y)
                xc = max(0, min(1, row['bbox_cx'] / row['image_width']))
                yc = max(0, min(1, row['bbox_cy'] / row['image_height']))
                
                # Normalized Width and Height 
                w  = max(0, min(1, row['bbox_w_norm']))
                h  = max(0, min(1, row['bbox_h_norm']))
                
                # Format: <class_id> <x_center> <y_center> <width> <height>
                f.write(f"{int(row['class_id'])} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
        
        count += 1
        if count % 500 == 0:
            # [Image of progress bar icon]
            print(f"  Processed {count}/{total_to_process} images in {split_name}...")

# ---------------------------------------------------------
# 5. EXECUTE
# ---------------------------------------------------------
DATASET_ROOT.mkdir(parents=True, exist_ok=True)

process_split(train_df, "train")
process_split(val_df, "val")

print("\n------------------------------------------------")
print("DATASET READY 🚀")
print(f"Final Dataset Root: {DATASET_ROOT.absolute()}")
print("------------------------------------------------")