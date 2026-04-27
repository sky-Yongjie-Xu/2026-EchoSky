# -*- coding: utf-8 -*-
# Standard library imports
import os
import torch
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from tqdm import tqdm
import click
from torchvision.models.segmentation import deeplabv3_resnet50

# Local module imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from modules.measurement.utils import (
    get_coordinates_from_dicom,
    calculate_weighted_centroids_with_meshgrid,
    ybr_to_rgb
)

# ==============================================
# Global Config
# ==============================================
SEGMENTATION_THRESHOLD = 0.0
DO_SIGMOID             = True

REGION_X0_SUBTAG                  = (0x0018, 0x6018)
REGION_Y0_SUBTAG                  = (0x0018, 0x601A)
REGION_X1_SUBTAG                  = (0x0018, 0x601C)
REGION_Y1_SUBTAG                  = (0x0018, 0x601E)
PHOTOMETRIC_INTERPRETATION_TAG    = (0x0028, 0x0004)
REGION_PHYSICAL_DELTA_Y_SUBTAG    = (0x0018, 0x602E)
ULTRASOUND_COLOR_DATA_PRESENT_TAG = (0x0028, 0x0014)
REFERENCE_LINE_TAG                = (0x0018, 0x6022)

# ==============================================
# Core Model Class
# ==============================================
class MVDopplerMeasurer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        BASE = os.path.dirname(os.path.abspath(__file__))
        self.weights_path = os.path.join(BASE, "../../weights/Doppler_models", "mvpeak_2c_weights.ckpt")

        if not os.path.exists(self.weights_path):
            alt = os.path.join(BASE, "weights/Doppler_models", "mvpeak_2c_weights.ckpt")
            if os.path.exists(alt):
                self.weights_path = alt

        print(f"[LOAD] MV E/A 模型: {self.weights_path}")

        # 加载模型
        weights = torch.load(self.weights_path, map_location=self.device, weights_only=True)
        self.backbone = deeplabv3_resnet50(num_classes=2)
        weights = {k.replace("m.", ""): v for k, v in weights.items()}
        self.backbone.load_state_dict(weights)
        self.backbone = self.backbone.to(self.device)
        self.backbone.eval()

    def forward_pass(self, inputs):
        logits = self.backbone(inputs)["out"]
        if DO_SIGMOID:
            logits = torch.sigmoid(logits)
        if SEGMENTATION_THRESHOLD is not None:
            logits[logits < SEGMENTATION_THRESHOLD] = 0.0

        logits_numpy = logits.squeeze().detach().cpu().numpy()
        logits_first = logits_numpy[1, :, :]
        max_val_first = logits_first.max()
        min_val_first = logits_first.min()
        logits_first = (logits_first - min_val_first) / (max_val_first - min_val_first)
        _, _, _, max_loc_first_channel = cv2.minMaxLoc(logits_first)

        logits_second = logits_numpy[0, :, :]
        max_val_second = logits_second.max()
        min_val_second = logits_second.min()
        logits_second = (logits_second - min_val_second) / (max_val_second - min_val_second)
        _, _, _, max_loc_second_channel = cv2.minMaxLoc(logits_second)

        combine_logit = logits_first + logits_second
        _, _, _, max_loc_combine = cv2.minMaxLoc(combine_logit)

        diff_maxloc_combine_first = np.sqrt(
            (max_loc_combine[0] - max_loc_first_channel[0])**2 +
            (max_loc_combine[1] - max_loc_first_channel[1])**2
        )
        diff_maxloc_combine_second = np.sqrt(
            (max_loc_combine[0] - max_loc_second_channel[0])**2 +
            (max_loc_combine[1] - max_loc_second_channel[1])**2
        )

        centroids_first, _ = calculate_weighted_centroids_with_meshgrid(logits_first)
        centroids_second, _ = calculate_weighted_centroids_with_meshgrid(logits_second)
        centroids, _ = calculate_weighted_centroids_with_meshgrid(combine_logit)

        distance_centroid_btw_maxlogits = {
            c: np.sqrt((max_loc_combine[0]-c[0])**2 + (max_loc_combine[1]-c[1])**2)
            for c in centroids
        }
        try:
            min_distance_coord = min(distance_centroid_btw_maxlogits, key=distance_centroid_btw_maxlogits.get)
        except ValueError:
            raise ValueError("min_distance_coord not found")

        if diff_maxloc_combine_second - diff_maxloc_combine_first > 15:
            pair_source = centroids_second
        elif diff_maxloc_combine_first - diff_maxloc_combine_second > 15:
            pair_source = centroids_first
        else:
            pair_source = centroids

        distance_btw_centroids = {
            c: np.sqrt((min_distance_coord[0]-c[0])**2 + (min_distance_coord[1]-c[1])**2)
            for c in pair_source
        }
        non_zero = {k: v for k, v in distance_btw_centroids.items() if v > 15}
        try:
            min_distance_paired_coord = min(non_zero, key=non_zero.get)
        except ValueError:
            raise ValueError("Pair point not found")

        point_x1, point_y1 = min_distance_coord
        point_x2, point_y2 = min_distance_paired_coord

        swapped = False
        if point_x1 > point_x2:
            point_x1, point_y1, point_x2, point_y2 = point_x2, point_y2, point_x1, point_y1
            swapped = True

        if abs(point_x1 - point_x2) > 300:
            raise ValueError("Distance between E/A points > 300 px")

        return point_x1, point_y1, point_x2, point_y2, swapped

# ==============================================
# Tool Functions
# ==============================================
def load_dicom_image(dicom_file):
    ds = pydicom.dcmread(dicom_file)
    input_image = ds.pixel_array
    meta = {}

    meta["PhotometricInterpretation"] = ds[PHOTOMETRIC_INTERPRETATION_TAG].value if PHOTOMETRIC_INTERPRETATION_TAG in ds else None
    meta["ultrasound_color_data_present"] = ds[ULTRASOUND_COLOR_DATA_PRESENT_TAG].value if ULTRASOUND_COLOR_DATA_PRESENT_TAG in ds else np.nan
    pi = meta["PhotometricInterpretation"]

    if pi == "MONOCHROME2":
        input_image = np.stack((input_image,) * 3, axis=-1)
    elif pi == "YBR_FULL_422" and len(input_image.shape) == 3:
        input_image = ybr_to_rgb(input_image)
        ecg_mask = np.logical_and(input_image[:, :, 1] > 200, input_image[:, :, 0] < 100)
        input_image[ecg_mask, :] = 0
    elif pi == "RGB":
        ecg_mask = np.logical_and(input_image[:, :, 1] > 200, input_image[:, :, 0] < 100)
        input_image[ecg_mask, :] = 0
    else:
        raise ValueError(f"Unsupported PI: {pi}")

    if len(input_image.shape) == 2:
        meta["height"], meta["width"] = input_image.shape
    else:
        meta["height"], meta["width"] = input_image.shape[0], input_image.shape[1]

    return input_image, ds, meta

def extract_doppler_tags(ds):
    doppler_region = get_coordinates_from_dicom(ds)[0]
    PhysicalDeltaY = abs(doppler_region[REGION_PHYSICAL_DELTA_Y_SUBTAG].value) if REGION_PHYSICAL_DELTA_Y_SUBTAG in doppler_region else None
    y0 = doppler_region[REGION_Y0_SUBTAG].value if REGION_Y0_SUBTAG in doppler_region else None
    y1 = doppler_region[REGION_Y1_SUBTAG].value if REGION_Y1_SUBTAG in doppler_region else None
    x0 = doppler_region[REGION_X0_SUBTAG].value if REGION_X0_SUBTAG in doppler_region else None
    x1 = doppler_region[REGION_X1_SUBTAG].value if REGION_X1_SUBTAG in doppler_region else None
    horizontal_y = doppler_region[REFERENCE_LINE_TAG].value if REFERENCE_LINE_TAG in doppler_region else 0
    return y0, y1, x0, x1, PhysicalDeltaY, horizontal_y

def save_jpg(input_image, x1, y1, x2, y2, y0, out_path):
    plt.figure(figsize=(8, 8))
    plt.imshow(input_image, cmap="gray")
    plt.scatter(x1, y1 + y0, color="red", s=20)
    plt.scatter(x2, y2 + y0, color="blue", s=20)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()

# ==============================================
# Unified Pipeline
# ==============================================
def run_pipeline(
    file_path=None,
    output_path=None,
    folders=None,
    output_path_folders=None
):
    if file_path is not None:
        MODE = "single"
    elif folders is not None:
        MODE = "folders"
    else:
        raise ValueError("Specify --file_path or --folders")

    measurer = MVDopplerMeasurer()
    device = measurer.device
    print(f"[✅] MV E/A 初始化完成 | 模式={MODE}")

    # --------------------------
    # Single Mode
    # --------------------------
    if MODE == "single":
        input_image, ds, meta = load_dicom_image(file_path)
        y0, y1, x0, x1, PhysicalDeltaY, horizontal_y = extract_doppler_tags(ds)

        if y0 < 340 or y0 > 350:
            raise ValueError(f"y0={y0} 超出范围 340-350")

        roi = input_image[342:, :, :]
        t = torch.tensor(roi).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        t = t.to(device)

        with torch.no_grad():
            px1, py1, px2, py2, swapped = measurer.forward_pass(t)

        E = round(abs((py1 - horizontal_y) * PhysicalDeltaY), 4)
        A = round(abs((py2 - horizontal_y) * PhysicalDeltaY), 4)
        ratio = round(E/A, 3) if A !=0 else float("nan")

        print(f"E={E} cm/s | A={A} cm/s | E/A={ratio}")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        save_jpg(input_image, px1, py1, px2, py2, y0, output_path)
        print(f"[✅] 保存: {output_path}")

    # --------------------------
    # Folders Batch Mode
    # --------------------------
    elif MODE == "folders":
        INPUT = folders
        OUTPUT = output_path_folders
        os.makedirs(OUTPUT, exist_ok=True)

        files = sorted([os.path.join(INPUT, f) for f in os.listdir(INPUT) if f.endswith(".dcm")])
        results = []
        ok_cnt = 0
        swap_cnt = 0
        err_cnt = 0

        for f in tqdm(files, desc="处理 MV E/A"):
            try:
                img, ds, meta = load_dicom_image(f)
                y0, y1, x0, x1, dY, hline = extract_doppler_tags(ds)

                if not (340 <= y0 <= 350):
                    err_cnt +=1
                    continue

                roi = img[342:, :, :]
                t = torch.tensor(roi).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                t = t.to(device)

                with torch.no_grad():
                    px1, py1, px2, py2, swapped = measurer.forward_pass(t)

                if swapped:
                    swap_cnt +=1

                E = round(abs((py1 - hline)*dY),4)
                A = round(abs((py2 - hline)*dY),4)
                ratio = round(E/A,3) if A !=0 else float("nan")

                if OUTPUT:
                    out = os.path.join(OUTPUT, os.path.basename(f).replace(".dcm", ".jpg"))
                    save_jpg(img, px1, py1, px2, py2, y0, out)

                results.append({
                    "file": f, "E": E, "A": A, "EA_ratio": ratio,
                    "xe": px1, "ye": py1, "xa": px2, "ya": py2
                })
                ok_cnt +=1

            except Exception as e:
                tqdm.write(f"❌ {os.path.basename(f)}: {str(e)}")
                err_cnt +=1

        pd.DataFrame(results).to_csv(os.path.join(OUTPUT, "metadata_mvpeak.csv"), index=False)
        print(f"[✅] 批量完成：OK={ok_cnt} 错误={err_cnt} 交换={swap_cnt}")

# ==============================================
# Click Command (兼容旧版)
# ==============================================
@click.command("doppler_mv_ea_measurement")
@click.option("--file_path", "-f", default=None, type=str)
@click.option("--output_path", "-o", default=None, type=str)
@click.option("--folders", "-dir", default=None, type=str)
@click.option("--output_path_folders", "-od", default=None, type=str)
def run(file_path, output_path, folders, output_path_folders):
    run_pipeline(
        file_path=file_path,
        output_path=output_path,
        folders=folders,
        output_path_folders=output_path_folders
    )

# ==============================================
# Register for Engine
# ==============================================
def register():
    return {
        "name": "doppler_mv_ea_measurement",
        "entry": run,
        "description": "二尖瓣多普勒 E/A 峰值测量 | 支持 single / folders"
    }

if __name__ == "__main__":
    run()