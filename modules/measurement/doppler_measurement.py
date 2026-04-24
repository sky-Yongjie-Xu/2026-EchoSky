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
from utils import (
    get_coordinates_from_dicom,
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
class DopplerMeasurer:
    def __init__(self, model_weights):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_weights = model_weights

        # 自动适配路径，防止找不到文件
        BASE = os.path.dirname(os.path.abspath(__file__))
        self.weights_path = os.path.join(BASE, "../../weights/Doppler_models", f"{model_weights}_weights.ckpt")

        if not os.path.exists(self.weights_path):
            alt = os.path.join(BASE, "weights/Doppler_models", f"{model_weights}_weights.ckpt")
            if os.path.exists(alt):
                self.weights_path = alt

        print(f"[LOAD] Doppler 模型: {self.weights_path}")

        # 加载模型
        weights = torch.load(self.weights_path, map_location=self.device, weights_only=True)
        self.backbone = deeplabv3_resnet50(num_classes=1)
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
        return logits

# ==============================================
# 工具函数
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
        from pydicom.pixel_data_handlers.util import convert_color_space
        input_image = convert_color_space(arr=input_image, current="YBR_FULL_422", desired="RGB")
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


def extract_doppler_region(ds):
    doppler_region = get_coordinates_from_dicom(ds)[0]
    conversion_factor = abs(doppler_region[REGION_PHYSICAL_DELTA_Y_SUBTAG].value) if REGION_PHYSICAL_DELTA_Y_SUBTAG in doppler_region else None
    y0 = doppler_region[REGION_Y0_SUBTAG].value if REGION_Y0_SUBTAG in doppler_region else None
    y1 = doppler_region[REGION_Y1_SUBTAG].value if REGION_Y1_SUBTAG in doppler_region else None
    x0 = doppler_region[REGION_X0_SUBTAG].value if REGION_X0_SUBTAG in doppler_region else None
    x1 = doppler_region[REGION_X1_SUBTAG].value if REGION_X1_SUBTAG in doppler_region else None
    horizontal_y = doppler_region[REFERENCE_LINE_TAG].value if REFERENCE_LINE_TAG in doppler_region else 0
    return conversion_factor, y0, y1, x0, x1, horizontal_y


def run_inference_on_image(input_image, y0, device, model):
    doppler_area = input_image[y0:, :, :]
    t = torch.tensor(doppler_area).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    t = t.to(device)

    with torch.no_grad():
        logit = model.forward_pass(t)

    max_val = logit.max().item()
    min_val = logit.min().item()
    logits_normalized = ((logit - min_val) / (max_val - min_val)).squeeze().cpu().numpy()
    max_coords = np.unravel_index(np.argmax(logits_normalized), logits_normalized.shape)

    X = int(max_coords[1])
    Y = int(max_coords[0])
    predicted_x = X
    predicted_y = Y + y0

    return logits_normalized, predicted_x, predicted_y, X, Y

# ==============================================
# 统一执行入口
# ==============================================
def run_pipeline(
    model_weights,
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

    model = DopplerMeasurer(model_weights)
    device = model.device
    print(f"[✅] Doppler 初始化完成 | 模式={MODE}")

    # --------------------------
    # Single
    # --------------------------
    if MODE == "single":
        input_image, ds, meta = load_dicom_image(file_path)
        cf, y0, y1, x0, x1, hline = extract_doppler_region(ds)
        logits_norm, px, py, X, Y = run_inference_on_image(input_image, y0, device, model)
        vel = round(cf * (py - (y0 + hline)), 2)
        print(f"[Result] 峰值速度: {vel} cm/s")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        doppler_area = input_image[y0:, :, :]
        heatmap = cv2.applyColorMap(np.uint8(255 * logits_norm), cv2.COLORMAP_MAGMA)
        overlay = cv2.addWeighted(doppler_area, 0.25, heatmap, 0.75, 0)
        cv2.imwrite(output_path.replace(".jpg", "_overlay.jpg"), overlay)
        cv2.imwrite(output_path.replace(".jpg", "_heatmap.jpg"), heatmap)

        cv2.circle(input_image, (px, py), 10, (135, 206, 235), -1)
        plt.figure(figsize=(4,4))
        plt.imshow(input_image)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        plt.close()
        print(f"[✅] 保存: {output_path}")

    # --------------------------
    # Folders 批量
    # --------------------------
    elif MODE == "folders":
        INPUT = folders
        OUTPUT = output_path_folders
        os.makedirs(OUTPUT, exist_ok=True)

        files = sorted([os.path.join(INPUT, f) for f in os.listdir(INPUT) if f.endswith(".dcm")])
        res = []
        ok = 0
        err = 0

        for f in tqdm(files, desc="多普勒批量处理"):
            try:
                img, ds, meta = load_dicom_image(f)
                cf, y0, y1, x0, x1, hline = extract_doppler_region(ds)

                if not (340 <= y0 <= 350):
                    tqdm.write(f"[跳过] {os.path.basename(f)} y0={y0}")
                    err +=1
                    continue

                logits_norm, px, py, X, Y = run_inference_on_image(img, y0, device, model)
                vel = round(cf * (py - (y0 + hline)), 2)

                if OUTPUT:
                    out = os.path.join(OUTPUT, os.path.basename(f).replace(".dcm", ".jpg"))
                    cp = img.copy()
                    cv2.circle(cp, (px, py), 10, (135,206,235), -1)
                    plt.figure(figsize=(4,4))
                    plt.imshow(cp)
                    plt.axis("off")
                    plt.savefig(out, bbox_inches="tight", pad_inches=0)
                    plt.close()

                res.append({
                    "file": f, "model": model_weights, "px": px, "py": py,
                    "velocity": vel, "deltaY": cf
                })
                ok +=1
            except Exception as e:
                tqdm.write(f"[错误] {f}: {e}")
                err +=1

        pd.DataFrame(res).to_csv(os.path.join(OUTPUT, f"metadata_{model_weights}.csv"), index=False)
        print(f"[✅] 批量完成: OK={ok} 错误={err}")

# ==============================================
# Click 命令
# ==============================================
@click.command("doppler_measurement")
@click.option("--model_weights", "-m", required=True, type=str) # choices=["avvmax", "trvmax", "mrvmax", "lvotvmax", "latevel", "medevel"]
@click.option("--file_path", "-f", default=None, type=str)
@click.option("--output_path", "-o", default=None, type=str)
@click.option("--folders", "-dir", default=None, type=str)
@click.option("--output_path_folders", "-od", default=None, type=str)
def run(
    model_weights,
    file_path,
    output_path,
    folders,
    output_path_folders
):
    run_pipeline(
        model_weights=model_weights,
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
        "name": "doppler_measurement",
        "entry": run,
        "description": "多普勒超声峰值速度测量 | 支持 single / folders"
    }

if __name__ == "__main__":
    run()