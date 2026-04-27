# -*- coding: utf-8 -*-
# Standard library imports
import os
import numpy as np

# Third-party library imports
import torch
import pandas as pd
import cv2
import pydicom
from tqdm import tqdm
import click
from torchvision.models.segmentation import deeplabv3_resnet50

# Local module imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from modules.measurement.utils import (
    segmentation_to_coordinates,
    process_video_with_diameter,
    get_coordinates_from_dicom,
    ybr_to_rgb
)

# ==============================================
# Global Config & DICOM Tags
# ==============================================
SEGMENTATION_THRESHOLD = 0.0
DO_SIGMOID = True

REGION_PHYSICAL_DELTA_X_SUBTAG    = (0x0018, 0x602C)
REGION_PHYSICAL_DELTA_Y_SUBTAG    = (0x0018, 0x602E)
PHOTOMETRIC_INTERPRETATION_TAG    = (0x0028, 0x0004)
ULTRASOUND_COLOR_DATA_PRESENT_TAG = (0x0028, 0x0014)

# ==============================================
# Core Inference Class
# ==============================================
class Seg2DMeasurer:
    def __init__(self, model_weights):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_weights = model_weights

        # 🔥 修复路径：从当前文件所在目录出发，自动找权重
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(BASE_DIR, "../../weights/2D_models", f"{model_weights}_weights.ckpt")

        if not os.path.exists(weights_path):
            alt_path = os.path.join(BASE_DIR, "weights/2D_models", f"{model_weights}_weights.ckpt")
            if os.path.exists(alt_path):
                weights_path = alt_path
            else:
                alt_path2 = os.path.join(BASE_DIR, f"{model_weights}_weights.ckpt")
                if os.path.exists(alt_path2):
                    weights_path = alt_path2

        print(f"[LOAD] 加载模型权重: {weights_path}")

        # 加载模型
        weights = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.backbone = deeplabv3_resnet50(num_classes=2)
        weights = {k.replace("m.", ""): v for k, v in weights.items()}
        self.backbone.load_state_dict(weights)
        self.backbone = self.backbone.to(self.device)
        self.backbone.eval()

    def forward_pass(self, inputs):
        with torch.no_grad():
            logits = self.backbone(inputs)["out"]
            if DO_SIGMOID:
                logits = torch.sigmoid(logits)
            if SEGMENTATION_THRESHOLD is not None:
                logits[logits < SEGMENTATION_THRESHOLD] = 0.0
        return segmentation_to_coordinates(logits, normalize=False, order="XY")

    def load_video_frames(self, video_file):
        frames = []
        meta = {
            "PhotometricInterpretation":     None,
            "ultrasound_color_data_present": None,
            "conversion_factor_X": None,
            "conversion_factor_Y": None,
            "ratio_height": 1.0,
            "ratio_width":  1.0,
        }

        if video_file.endswith(".avi"):
            cap = cv2.VideoCapture(video_file)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()

        elif video_file.endswith(".dcm"):
            ds = pydicom.dcmread(video_file)
            meta["ultrasound_color_data_present"] = (
                ds[ULTRASOUND_COLOR_DATA_PRESENT_TAG].value
                if ULTRASOUND_COLOR_DATA_PRESENT_TAG in ds else np.nan
            )
            meta["PhotometricInterpretation"] = (
                ds[PHOTOMETRIC_INTERPRETATION_TAG].value
                if PHOTOMETRIC_INTERPRETATION_TAG in ds else None
            )

            pixel_array = ds.pixel_array
            height, width = pixel_array.shape[1], pixel_array.shape[2]
            meta["ratio_height"] = height / 480
            meta["ratio_width"]  = width  / 640

            regions = get_coordinates_from_dicom(ds)
            doppler_region = regions[0] if regions else None
            if doppler_region is not None:
                if REGION_PHYSICAL_DELTA_X_SUBTAG in doppler_region:
                    meta["conversion_factor_X"] = abs(doppler_region[REGION_PHYSICAL_DELTA_X_SUBTAG].value)
                if REGION_PHYSICAL_DELTA_Y_SUBTAG in doppler_region:
                    meta["conversion_factor_Y"] = abs(doppler_region[REGION_PHYSICAL_DELTA_Y_SUBTAG].value)

            for frame in pixel_array:
                if ds.PhotometricInterpretation == "YBR_FULL_422":
                    frame = ybr_to_rgb(frame)
                frames.append(cv2.resize(frame, (640, 480)))
        else:
            raise ValueError(f"Unsupported file type: {video_file}")
        return frames, meta

    @staticmethod
    def make_annotated_frame(frame_tensor, prediction, dot_radius=5, color=(235, 206, 135)):
        frame = frame_tensor.permute(1, 2, 0).cpu().numpy()
        frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        x1, y1 = int(prediction[0][0]), int(prediction[0][1])
        x2, y2 = int(prediction[1][0]), int(prediction[1][1])
        cv2.circle(frame, (x1, y1), dot_radius, color, -1)
        cv2.circle(frame, (x2, y2), dot_radius, color, -1)
        cv2.line(frame, (x1, y1), (x2, y2), color, 2)
        return frame, x1, y1, x2, y2

    @staticmethod
    def compute_diameter(x1, y1, x2, y2, ratio, conv_x, conv_y):
        if conv_x is None or conv_y is None:
            return None
        delta_x = abs(x2 - x1) * ratio
        delta_y = abs(y2 - y1) * ratio
        return float(np.sqrt((delta_x * conv_x) ** 2 + (delta_y * conv_y) ** 2))

# ==============================================
# Unified Pipeline
# ==============================================
def run_pipeline(
    model_weights,
    file_path=None,
    output_path=None,
    phase_estimate=False,
    folders=None,
    output_path_folders=None,
    manifest_with_frame=None
):
    if file_path is not None:
        MODE = "single"
    elif folders is not None and manifest_with_frame is not None:
        MODE = "manifest"
    elif folders is not None:
        MODE = "folders"
    else:
        raise ValueError("必须指定 --file_path / --folders / --manifest_with_frame")

    measurer = Seg2DMeasurer(model_weights)
    print(f"[✅] 模型加载完成 | 测量: {model_weights} | 模式: {MODE}")

    # --------------------------
    # 单文件模式
    # --------------------------
    if MODE == "single":
        VIDEO_FILE = file_path
        OUTPUT_FILE = output_path
        frames, meta = measurer.load_video_frames(VIDEO_FILE)
        input_tensor = torch.tensor(np.array(frames)).float() / 255.0
        input_tensor = input_tensor.to(measurer.device).permute(0, 3, 1, 2)

        predictions = []
        for i in range(input_tensor.shape[0]):
            predictions.append(measurer.forward_pass(input_tensor[i].unsqueeze(0)))
        predictions = torch.cat(predictions, dim=0).cpu().numpy()

        os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_FILE)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*("XVID" if OUTPUT_FILE.endswith(".avi") else "mp4v"))
        h, w = input_tensor.shape[-2], input_tensor.shape[-1]
        out_video = cv2.VideoWriter(OUTPUT_FILE, fourcc, 30, (w, h))

        frame_nums, x1s, y1s, x2s, y2s = [], [], [], [], []
        for i, (frame_t, pred) in enumerate(zip(input_tensor, predictions)):
            annotated, x1, y1, x2, y2 = measurer.make_annotated_frame(frame_t, pred)
            out_video.write(annotated)
            frame_nums.append(i)
            x1s.append(x1); y1s.append(y1)
            x2s.append(x2); y2s.append(y2)
        out_video.release()

        df = pd.DataFrame({
            "frame_number": frame_nums,
            "pred_x1": x1s, "pred_y1": y1s,
            "pred_x2": x2s, "pred_y2": y2s
        })
        csv_path = OUTPUT_FILE.rsplit(".", 1)[0] + ".csv"
        df.to_csv(csv_path, index=False)
        print(f"[✅] 输出视频: {OUTPUT_FILE}")
        print(f"[✅] 输出坐标: {csv_path}")

    # --------------------------
    # 批量文件夹模式（你要的！）
    # --------------------------
    elif MODE == "folders":
        INPUT_FOLDER   = folders
        OUTPUT_FOLDER  = output_path_folders
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

        VIDEO_FILES = sorted([
            os.path.join(INPUT_FOLDER, f)
            for f in os.listdir(INPUT_FOLDER)
            if f.endswith(".dcm") or f.endswith(".avi")
        ])
        print(f"[📂] 找到 {len(VIDEO_FILES)} 个文件")

        results_all = []
        for VIDEO_FILE in tqdm(VIDEO_FILES, desc="处理中"):
            try:
                frames, meta = measurer.load_video_frames(VIDEO_FILE)
                conv_x = meta["conversion_factor_X"]
                conv_y = meta["conversion_factor_Y"]
                ratio  = meta["ratio_height"]

                input_tensor = torch.tensor(np.array(frames)).float() / 255.0
                input_tensor = input_tensor.to(measurer.device).permute(0, 3, 1, 2)

                predictions = []
                for i in range(input_tensor.shape[0]):
                    predictions.append(measurer.forward_pass(input_tensor[i].unsqueeze(0)))
                predictions = torch.cat(predictions, dim=0).cpu().numpy()

                out_path = os.path.join(OUTPUT_FOLDER, os.path.basename(VIDEO_FILE) + "_result.avi")
                h, w = input_tensor.shape[-2], input_tensor.shape[-1]
                out_avi = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"XVID"), 30, (w, h))

                for i, (frame_t, pred) in enumerate(zip(input_tensor, predictions)):
                    annotated, x1, y1, x2, y2 = measurer.make_annotated_frame(frame_t, pred)
                    out_avi.write(annotated)
                    results_all.append({
                        "filename": VIDEO_FILE,
                        "frame_number": i,
                        "measurement": model_weights,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "diameter": measurer.compute_diameter(x1,y1,x2,y2,ratio,conv_x,conv_y)
                    })
                out_avi.release()
                tqdm.write(f"✅ {os.path.basename(VIDEO_FILE)} -> {out_path}")
            except Exception as e:
                tqdm.write(f"❌ {os.path.basename(VIDEO_FILE)} 失败: {str(e)}")

        csv_path = os.path.join(OUTPUT_FOLDER, f"result_{model_weights}.csv")
        pd.DataFrame(results_all).to_csv(csv_path, index=False)
        print(f"[✅] 批量完成！结果: {csv_path}")

# ==============================================
# Click Command
# ==============================================
@click.command("b_mode_linear_measurement")
@click.option("--model_weights", "-m", required=True, type=str) # choices=["ivs", "lvid", "lvpw", "aorta", "aortic_root", "la", "rv_base", "pa", "ivc"]
@click.option("--file_path", "-f", default=None, type=str)
@click.option("--output_path", "-o", default=None, type=str)
@click.option("--phase_estimate", is_flag=True, default=False)
@click.option("--folders", "-dir", default=None, type=str)
@click.option("--output_path_folders", "-od", default=None, type=str)
@click.option("--manifest_with_frame", "-csv", default=None, type=str)
def run(
    model_weights,
    file_path,
    output_path,
    phase_estimate,
    folders,
    output_path_folders,
    manifest_with_frame
):
    run_pipeline(
        model_weights=model_weights,
        file_path=file_path,
        output_path=output_path,
        phase_estimate=phase_estimate,
        folders=folders,
        output_path_folders=output_path_folders,
        manifest_with_frame=manifest_with_frame
    )

# ==============================================
# Register for Engine
# ==============================================
def register():
    return {
        "name": "b_mode_linear_measurement",
        "entry": run,
        "description": "心脏2D结构分割测量 | IVS/LVID/LVPW/Aorta/LA/RV/PA/IVC"
    }

if __name__ == "__main__":
    run()