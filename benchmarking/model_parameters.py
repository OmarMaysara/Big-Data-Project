# parameters.py
from ultralytics import YOLO
import json
import os
import multiprocessing
import numbers

# -------------------------------
# Config - edit paths as needed
# -------------------------------
model_path = r"F:\BDD Training\runs\detect\Experiment4_yolo26n_cos_lr=True\train\weights\best.pt"
data_path  = r"F:\Datasets\bdd100k\bdd100k.yaml" #BDD100k
save_dir   = r"F:\BDD Training\runs\detect\validation_metrics"
os.makedirs(save_dir, exist_ok=True)

# -------------------------------
# Helpers
# -------------------------------
def get_attr(obj, *attrs):
    """Return first existing attribute name from obj without evaluating arrays' truth value."""
    for a in attrs:
        if hasattr(obj, a):
            return getattr(obj, a)
    return None

def to_float(x):
    """Convert torch tensor / numpy scalar / python number -> python float (or None)."""
    if x is None:
        return None
    # torch
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return float(x.detach().cpu().item())
    except Exception:
        pass
    # numpy
    try:
        import numpy as np
        if isinstance(x, (np.generic,)) and np.size(x) == 1:
            return float(x.item())
        if isinstance(x, (np.ndarray,)) and x.size == 1:
            return float(x.item())
    except Exception:
        pass
    # python number
    if isinstance(x, numbers.Number):
        return float(x)
    # fallback
    try:
        return float(x)
    except Exception:
        return None

def list_to_floats(x):
    """Convert list-like / tensor / ndarray to list of floats."""
    if x is None:
        return []
    try:
        import torch
        if isinstance(x, torch.Tensor):
            # convert tensor to python list first
            return [to_float(v) for v in x.detach().cpu().tolist()]
    except Exception:
        pass
    try:
        import numpy as np
        if isinstance(x, (np.ndarray,)):
            return [to_float(v) for v in x.tolist()]
    except Exception:
        pass
    # If it's already a list or iterable
    try:
        return [to_float(v) for v in list(x)]
    except Exception:
        return []

# -------------------------------
# Validate and extract metrics
# -------------------------------
def validate():
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # Run validation. workers=0 avoids multiprocessing spawn issues on Windows.
    print("Running validation (this may take a while)...")
    results = model.val(data=data_path, save_json=True, plots=True, workers=0, imgsz=640    )

    output = {"model": os.path.abspath(model_path), "saved_plots_dir": None, "predictions_json": None}

    # Locate latest runs/detect/val* folder and predictions.json
    try:
        runs_dir = os.path.join("runs", "detect")
        if os.path.isdir(runs_dir):
            val_dirs = [d for d in os.listdir(runs_dir) if d.startswith("val")]
            if val_dirs:
                val_dirs_sorted = sorted(val_dirs, key=lambda n: os.path.getmtime(os.path.join(runs_dir, n)))
                last = val_dirs_sorted[-1]
                last_path = os.path.join(runs_dir, last)
                output["saved_plots_dir"] = os.path.abspath(last_path)
                pj = os.path.join(last_path, "predictions.json")
                if os.path.exists(pj):
                    output["predictions_json"] = os.path.abspath(pj)
    except Exception:
        pass

    # Extract box metrics safely
    box = get_attr(results, "box")

    if box is not None:
        # safe retrieval without using 'or' on arrays
        p_list    = get_attr(box, "p", "P")
        r_list    = get_attr(box, "r", "R")
        f1_list   = get_attr(box, "f1")
        ap50_list = get_attr(box, "ap50", "ap_50", "ap_0.5")
        ap_list   = get_attr(box, "ap", "maps")

        map50     = get_attr(box, "map50", "map_50")
        map50_95  = get_attr(box, "map", "map50_95")

        # compute global as means if scalars not present
        precision_global = to_float(get_attr(box, "P_mean")) if get_attr(box, "P_mean") is not None else (
            (sum(list_to_floats(p_list)) / len(list_to_floats(p_list))) if list_to_floats(p_list) else None
        )
        recall_global = to_float(get_attr(box, "R_mean")) if get_attr(box, "R_mean") is not None else (
            (sum(list_to_floats(r_list)) / len(list_to_floats(r_list))) if list_to_floats(r_list) else None
        )
        f1_global = to_float(get_attr(box, "f1_mean")) if get_attr(box, "f1_mean") is not None else (
            (sum(list_to_floats(f1_list)) / len(list_to_floats(f1_list))) if list_to_floats(f1_list) else None
        )

        output["metrics_global"] = {
            "precision": to_float(precision_global),
            "recall": to_float(recall_global),
            "f1_score": to_float(f1_global),
            "mAP_0.5": to_float(map50),
            "mAP_0.5_0.95": to_float(map50_95),
        }

        output["metrics_per_class"] = {
            "class_names": getattr(results, "names", None),
            "precision": list_to_floats(p_list),
            "recall": list_to_floats(r_list),
            "f1_score": list_to_floats(f1_list),
            "ap50": list_to_floats(ap50_list),
            "ap": list_to_floats(ap_list),
        }
    else:
        # fallback for other ultralytics versions
        if hasattr(results, "results_dict"):
            output["metrics_global"] = results.results_dict
        else:
            output["metrics_global"] = {}

    # Confusion matrix
    cm = get_attr(results, "confusion_matrix") or (get_attr(box, "confusion_matrix") if box is not None else None)
    if cm is not None:
        try:
            output["confusion_matrix"] = cm.tolist()
        except Exception:
            output["confusion_matrix"] = str(cm)

    # Save outputs
    json_path = os.path.join(save_dir, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    txt_path = os.path.join(save_dir, "metrics.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(output, indent=4, ensure_ascii=False))

    print(f"\nMetrics saved to:\n{json_path}\n{txt_path}")
    if output.get("saved_plots_dir"):
        print(f"Validation run directory (plots/predictions): {output['saved_plots_dir']}")
    if output.get("predictions_json"):
        print(f"Predictions JSON: {output['predictions_json']}")

# -------------------------------
# Main guard (required on Windows)
# -------------------------------
if __name__ == "_main_":
    multiprocessing.freeze_support()
validate()