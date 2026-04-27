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

# DICOM tags
REGION_X0_SUBTAG                  = (0x0018, 0x6018)
REGION_Y0_SUBTAG                  = (0x0018, 0x601A)
REGION_X1_SUBTAG                  = (0x0018, 0x601C)
REGION_Y1_SUBTAG                  = (0x0018, 0x601E)
PHOTOMETRIC_INTERPRETATION_TAG    = (0x0028, 0x0004)
REGION_PHYSICAL_DELTA_X_SUBTAG    = (0x0018, 0x602C)
REGION_PHYSICAL_DELTA_Y_SUBTAG    = (0x0018, 0x602E)
ULTRASOUND_COLOR_DATA_PRESENT_TAG = (0x0028, 0x0014)

# ==============================================
# Core Model Class
# ==============================================
class TAPSEMeasurer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        BASE = os.path.dirname(os.path.abspath(__file__))
        
        # 自动搜索权重路径，防止找不到文件
        self.weights_path = os.path.join(BASE, "../../weights/Doppler_models", "tapse_2c_weights.ckpt")
        if not os.path.exists(self.weights_path):
            alt_path = os.path.join(BASE, "weights/Doppler_models", "tapse_2c_weights.ckpt")
            if os.path.exists(alt_path):
                self.weights_path = alt_path

        print(f"[LOAD] TAPSE 模型权重: {self.weights_path}")

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
        rng = logits_first.max() - logits_first.min()
        logits_first = (logits_first - logits_first.min()) / rng if rng > 0 else logits_first
        _, _, _, max_loc_first_channel = cv2.minMaxLoc(logits_first)

        logits_second = logits_numpy[0, :, :]
        rng = logits_second.max() - logits_second.min()
        logits_second = (logits_second - logits_second.min()) / rng if rng > 0 else logits_second
        _, _, _, max_loc_second_channel = cv2.minMaxLoc(logits_second)

        combine_logit = logits_first + logits_second
        _, _, _, max_loc_combine = cv2.minMaxLoc(combine_logit)

        diff_first  = np.sqrt((max_loc_combine[0]-max_loc_first_channel[0])**2  + (max_loc_combine[1]-max_loc_first_channel[1])**2)
        diff_second = np.sqrt((max_loc_combine[0]-max_loc_second_channel[0])**2 + (max_loc_combine[1]-max_loc_second_channel[1])**2)

        centroids_first,  _ = calculate_weighted_centroids_with_meshgrid(logits_first)
        centroids_second, _ = calculate_weighted_centroids_with_meshgrid(logits_second)
        centroids,        _ = calculate_weighted_centroids_with_meshgrid(combine_logit)

        d_btw_max = {c: np.sqrt((max_loc_combine[0]-c[0])**2 + (max_loc_combine[1]-c[1])**2) for c in centroids}
        try:
            min_coord = min(d_btw_max, key=d_btw_max.get)
        except ValueError:
            raise ValueError("min_distance_coord not found")

        if diff_second - diff_first > 15:
            pair_source = centroids_second
        elif diff_first - diff_second > 15:
            pair_source = centroids_first
        else:
            pair_source = centroids

        d_btw = {c: np.sqrt((min_coord[0]-c[0])**2 + (min_coord[1]-c[1])**2) for c in pair_source}
        non_zero = {k: v for k, v in d_btw.items() if v > 15}
        try:
            paired_coord = min(non_zero, key=non_zero.get)
        except ValueError:
            raise ValueError("Pair point not found")

        point_x1, point_y1 = min_coord
        point_x2, point_y2 = paired_coord

        if point_x1 > point_x2:
            point_x1, point_y1, point_x2, point_y2 = point_x2, point_y2, point_x1, point_y1

        return point_x1, point_y1, point_x2, point_y2

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

    return input_image, ds, meta

def extract_doppler_tags(ds):
    doppler_region = get_coordinates_from_dicom(ds)[0]
    PhysicalDeltaX = abs(doppler_region[REGION_PHYSICAL_DELTA_X_SUBTAG].value) if REGION_PHYSICAL_DELTA_X_SUBTAG in doppler_region else None
    PhysicalDeltaY = abs(doppler_region[REGION_PHYSICAL_DELTA_Y_SUBTAG].value) if REGION_PHYSICAL_DELTA_Y_SUBTAG in doppler_region else None
    y0 = doppler_region[REGION_Y0_SUBTAG].value if REGION_Y0_SUBTAG in doppler_region else None
    y1 = doppler_region[REGION_Y1_SUBTAG].value if REGION_Y1_SUBTAG in doppler_region else None
    x0 = doppler_region[REGION_X0_SUBTAG].value if REGION_X0_SUBTAG in doppler_region else None
    x1 = doppler_region[REGION_X1_SUBTAG].value if REGION_X1_SUBTAG in doppler_region else None
    return y0, y1, x0, x1, PhysicalDeltaX, PhysicalDeltaY

def compute_tapse(x1, y1, x2, y2, delta_x, delta_y):
    return round(float(np.sqrt(
        (abs(x1 - x2) * delta_x) ** 2 +
        (abs(y1 - y2) * delta_y) ** 2
    )), 2)

def save_jpg(input_image, x1, y1, x2, y2, y0, out_path):
    img_copy = input_image.copy()
    cv2.line(img_copy, (x1, y1 + y0), (x2, y2 + y0), (255, 0, 0), 2)
    plt.figure(figsize=(8, 8))
    plt.imshow(img_copy, cmap="gray")
    plt.scatter(x1, y1 + y0, color="red", s=20)
    plt.scatter(x2, y2 + y0, color="red", s=20)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
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
    # 模式判断
    if file_path is not None:
        MODE = "single"
    elif folders is not None:
        MODE = "folders"
    else:
        raise ValueError("必须指定 --file_path 或 --folders")

    # 初始化模型
    measurer = TAPSEMeasurer()
    device = measurer.device
    print(f"[✅] TAPSE 初始化完成 | 运行模式: {MODE}")

    # --------------------------
    # 单文件模式
    # --------------------------
    if MODE == "single":
        input_image, ds, meta = load_dicom_image(file_path)
        y0, y1, x0, x1, delta_x, delta_y = extract_doppler_tags(ds)

        # 截取ROI
        doppler_roi = input_image[y0:, :, :]
        t = torch.tensor(doppler_roi).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        t = t.to(device)

        with torch.no_grad():
            px1, py1, px2, py2 = measurer.forward_pass(t)

        # 计算TAPSE
        tapse_val = compute_tapse(px1, py1, px2, py2, delta_x, delta_y)
        print(f"[结果] TAPSE = {tapse_val} cm")

        # 保存图片
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        save_jpg(input_image, px1, py1, px2, py2, y0, output_path)
        print(f"[✅] 结果已保存: {output_path}")

    # --------------------------
    # 批量文件夹模式
    # --------------------------
    elif MODE == "folders":
        INPUT = folders
        OUTPUT = output_path_folders
        os.makedirs(OUTPUT, exist_ok=True)

        # 获取所有dcm文件
        dicom_files = sorted([os.path.join(INPUT, f) for f in os.listdir(INPUT) if f.endswith(".dcm")])
        results = []
        ok_cnt = 0
        err_cnt = 0

        for f in tqdm(dicom_files, desc="TAPSE 处理中"):
            try:
                img, ds, meta = load_dicom_image(f)
                y0, y1, x0, x1, delta_x, delta_y = extract_doppler_tags(ds)

                # 过滤异常y0
                if not (340 <= y0 <= 350):
                    tqdm.write(f"[跳过] {os.path.basename(f)}: y0 超出范围")
                    err_cnt += 1
                    continue

                # 推理
                doppler_roi = img[y0:, :, :]
                t = torch.tensor(doppler_roi).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                t = t.to(device)

                with torch.no_grad():
                    px1, py1, px2, py2 = measurer.forward_pass(t)

                # 点位顺序检查
                if (px1 < px2) and (py1 < py2):
                    tqdm.write(f"[跳过] {os.path.basename(f)}: 点位顺序异常")
                    err_cnt += 1
                    continue

                # 计算结果
                tapse_val = compute_tapse(px1, py1, px2, py2, delta_x, delta_y)

                # 保存图片
                if OUTPUT:
                    out_jpg = os.path.join(OUTPUT, os.path.basename(f).replace(".dcm", ".jpg"))
                    save_jpg(img, px1, py1, px2, py2, y0, out_jpg)

                # 记录结果
                results.append({
                    "filename": f,
                    "TAPSE": tapse_val,
                    "x1": px1, "y1": py1,
                    "x2": px2, "y2": py2
                })
                ok_cnt += 1
                tqdm.write(f"[OK] {os.path.basename(f)} | TAPSE={tapse_val} cm")

            except Exception as e:
                tqdm.write(f"[错误] {os.path.basename(f)}: {str(e)}")
                err_cnt += 1

        # 保存CSV
        csv_path = os.path.join(OUTPUT, "metadata_tapse.csv")
        pd.DataFrame(results).to_csv(csv_path, index=False)
        print(f"\n[✅] 批量处理完成：成功={ok_cnt} | 失败={err_cnt}")
        print(f"[✅] 结果表格: {csv_path}")

# ==============================================
# Click 命令（兼容旧版）
# ==============================================
@click.command("doppler_tapse_measurement")
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
# 引擎注册
# ==============================================
def register():
    return {
        "name": "doppler_tapse_measurement",
        "entry": run,
        "description": "TAPSE 三尖瓣环收缩期位移测量 | single + folders 批量"
    }

if __name__ == "__main__":
    run()