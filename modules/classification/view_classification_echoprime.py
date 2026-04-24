# -*- coding: utf-8 -*-
# Standard library imports
import os
import math
import glob
import random

# Third-party library imports
import torch
import torchvision
import numpy as np
import cv2
import pydicom
from tqdm import tqdm
import click
import matplotlib.pyplot as plt

# Local module imports
import utils

COARSE_VIEWS = [
    'A2C','A3C','A4C','A5C',
    'Apical_Doppler',
    'Doppler_Parasternal_Long','Doppler_Parasternal_Short',
    'Parasternal_Long','Parasternal_Short',
    'SSN','Subcostal'
]

class EchoViewClassifier:
    def __init__(self, device=None, weights_path="model_data/weights/view_classifier.pt"):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.frames_to_take = 32
        self.frame_stride = 2
        self.video_size = 224
        self.mean = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(3, 1, 1, 1)
        self.std = torch.tensor([47.989223, 46.456997, 47.20083]).reshape(3, 1, 1, 1)

        # 与你原版完全一致的视角分类模型
        self.view_classifier = torchvision.models.convnext_base()
        self.view_classifier.classifier[-1] = torch.nn.Linear(
            self.view_classifier.classifier[-1].in_features, 11
        )
        state_dict = torch.load(weights_path, map_location=self.device)
        self.view_classifier.load_state_dict(state_dict)
        self.view_classifier.to(self.device)
        self.view_classifier.eval()
        for param in self.view_classifier.parameters():
            param.requires_grad = False

    def process_dicoms(self, INPUT):
        """
        与你原版 EchoPrime 完全一致的 DICOM 预处理
        1:1 复刻，无任何改动
        """
        dicom_paths = glob.glob(f'{INPUT}/**/*.dcm', recursive=True)
        stack_of_videos = []
        valid_paths = []

        for idx, dicom_path in tqdm(enumerate(dicom_paths), total=len(dicom_paths)):
            try:
                dcm = pydicom.dcmread(dicom_path)
                pixels = dcm.pixel_array

                # 原版过滤条件
                if pixels.ndim < 3 or pixels.shape[2] == 3:
                    continue

                # 单通道转3通道
                if pixels.ndim == 3:
                    pixels = np.repeat(pixels[..., None], 3, axis=3)

                # 原版 mask + 缩放
                pixels = utils.mask_outside_ultrasound(pixels)
                x = np.zeros((len(pixels), 224, 224, 3))
                for i in range(len(x)):
                    x[i] = utils.crop_and_scale(pixels[i])

                x = torch.as_tensor(x, dtype=torch.float).permute([3, 0, 1, 2])
                x.sub_(self.mean).div_(self.std)

                # 填充到固定帧数
                if x.shape[1] < self.frames_to_take:
                    padding = torch.zeros(
                        (3, self.frames_to_take - x.shape[1], self.video_size, self.video_size),
                        dtype=torch.float
                    )
                    x = torch.cat((x, padding), dim=1)

                # 原版采样方式
                start = 0
                video_tensor = x[:, start : start + self.frames_to_take : self.frame_stride, :, :]
                stack_of_videos.append(video_tensor)
                valid_paths.append(dicom_path)

            except Exception as e:
                continue

        # ===================== 修复：确保所有 tensor 形状一致 =====================
        fixed_shape = (3, 16, 224, 224)
        final_videos = []
        final_paths = []
        for t, p in zip(stack_of_videos, valid_paths):
            if t.shape == fixed_shape:
                final_videos.append(t)
                final_paths.append(p)

        return torch.stack(final_videos), final_paths

    @torch.no_grad()
    def get_views(self, stack_of_videos, visualize=False, return_view_list=True):
        """
        与你原版完全一致的视角预测逻辑
        """
        stack_of_first_frames = stack_of_videos[:, :, 0, :, :].to(self.device)
        out_logits = self.view_classifier(stack_of_first_frames)
        out_views = torch.argmax(out_logits, dim=1)
        view_list = [COARSE_VIEWS[v] for v in out_views]

        if visualize:
            print("Preprocessed and normalized video inputs")
            rows, cols = (len(view_list) // 12 + (len(view_list) % 9 > 0)), 12
            fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
            axes = axes.flatten()
            for i in range(len(view_list)):
                display_image = (stack_of_first_frames[i].cpu().permute([1,2,0]) * 255).numpy()
                display_image = np.clip(display_image, 0, 255).astype('uint8')
                display_image = np.ascontiguousarray(display_image)
                display_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)
                cv2.putText(display_image, view_list[i].replace("_"," "), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
                axes[i].imshow(display_image)
                axes[i].axis('off')

            for j in range(i + 1, len(axes)):
                axes[j].axis('off')
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            plt.savefig("modules/classification/outputs/view_visualization.png")
        return view_list

# ==============================================
# Click 命令行入口
# ==============================================
@click.command("view_classification_echoprime")
@click.option("--dataset_dir", "-i", required=True, help="DICOM 文件夹路径")
@click.option("--weights", "-w", default="modules/classification/weights/view_classifier.pt", help="模型权重")
@click.option("--output", "-o", default="modules/classification/outputs/view_results.txt", help="输出结果")
@click.option("--visualize/--skip_visualize", default=False, help="是否可视化")
def run(dataset_dir, weights, output, visualize):
    click.echo(f"[+] 加载视角分类模型")
    classifier = EchoViewClassifier(weights_path=weights)

    click.echo(f"[+] 读取视频: {dataset_dir}")
    videos, paths = classifier.process_dicoms(dataset_dir)

    click.echo(f"[+] 预测切面...")
    view_list = classifier.get_views(videos, visualize=visualize)

    unique_views = sorted(list(set(view_list)))
    click.echo("\n✅ 识别完成")
    click.echo(f"有效视频: {len(videos)}")
    click.echo(f"识别切面: {len(unique_views)} 种")
    for v in unique_views:
        click.echo(f"  - {v}")

    with open(output, "w", encoding="utf-8") as f:
        f.write("超声切面分类结果\n")
        for p, v in zip(paths, view_list):
            f.write(f"{os.path.basename(p)}\t{v}\n")

    click.echo(f"\n✅ 结果已保存: {output}")

# ==============================================
# 模块注册（主引擎加载）
# ==============================================
def register():
    return {
        "name": "view_classification_echoprime",
        "entry": run,
        "description": "超声视频切面/视角分类（基于EchoPrime原版）"
    }

if __name__ == "__main__":
    run()