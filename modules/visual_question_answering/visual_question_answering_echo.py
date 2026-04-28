# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import click
import sys
import glob
import cv2
import numpy as np
import pydicom
from pathlib import Path
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType

# 自动加入项目根目录，确保导入正常
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ====================== 量化配置 ======================
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_use_double_quant=True,
    bnb_8bit_quant_type="int8"
)
model_dtype = torch.float16

# ====================== EchoGemma 模型类 ======================
class EchoGemmaVQA(nn.Module):
    def __init__(self, emb_dim=512, device=torch.device('cuda')):
        super().__init__()
        self.device = device

        # ---------- Tokenizer & MedGemma ----------
        self.tokenizer = AutoTokenizer.from_pretrained(
            "modules/report_generation/medgemma-1.5-4b-it", use_fast=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.medgemma = AutoModelForCausalLM.from_pretrained(
            "modules/report_generation/medgemma-1.5-4b-it",
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
            dtype=model_dtype
        )

        # LoRA
        lora_target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=lora_target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        self.medgemma = get_peft_model(self.medgemma, lora_config)

        # ---------- 视频编码器 ----------
        self.echo_encoder = torchvision.models.video.mvit_v2_s()
        self.echo_encoder.head[-1] = nn.Linear(
            self.echo_encoder.head[-1].in_features, 512
        )

        # ---------- 视图分类器 ----------
        self.view_classifier = torchvision.models.convnext_base()
        self.view_classifier.classifier[-1] = nn.Linear(
            self.view_classifier.classifier[-1].in_features, 11
        )

        # ---------- 投影层 ----------
        self.visual_projection = nn.Linear(emb_dim + 11, 2560, dtype=torch.float16)

        # ---------- 视频预处理 ----------
        self.frames_to_take = 16
        self.frame_stride = 2
        self.video_size = 224
        self.mean = torch.tensor([29.1106, 28.0768, 29.0964]).reshape(3,1,1,1).half().to(self.device)
        self.std = torch.tensor([47.9892, 46.4570, 47.2008]).reshape(3,1,1,1).half().to(self.device)

        # ---------- 加载权重 ----------
        checkpoint = torch.load(
            "modules/report_generation/weights/echogemma.pt",
            map_location="cpu"
        )
        self.load_state_dict(checkpoint, strict=False)
        self.to(device).half()
        self.eval()

    # ====================== 处理视频（支持普通视频 + DICOM） ======================
    def process_video_folder(self, video_dir):
        video_paths = glob.glob(f"{video_dir}/**/*.mp4", recursive=True) + \
                      glob.glob(f"{video_dir}/**/*.avi", recursive=True) + \
                      glob.glob(f"{video_dir}/**/*.dcm", recursive=True)

        videos = []
        for path in tqdm(video_paths, desc="Processing videos"):
            try:
                if path.endswith(".dcm"):
                    video = self._load_dicom(path)
                else:
                    video = self._load_video(path)

                if video is None:
                    continue

                videos.append(video)
            except Exception as e:
                print(f"Skip {path}: {e}")

        if not videos:
            raise ValueError("No valid videos found")

        return torch.stack(videos).to(self.device)

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while len(frames) < self.frames_to_take:
            ret, f = cap.read()
            if not ret:
                break
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            f = self.crop_and_scale(f)
            frames.append(f)
        cap.release()

        if len(frames) == 0:
            return None

        x = np.array(frames, dtype=np.float16)
        x = torch.from_numpy(x).permute(3,0,1,2)
        x = x[:, ::self.frame_stride][:, :self.frames_to_take]
        x = (x.half() - self.mean.squeeze()) / self.std.squeeze()

        if x.shape[1] < self.frames_to_take:
            pad = torch.zeros(3, self.frames_to_take - x.shape[1], 224,224, dtype=torch.half)
            x = torch.cat([x, pad], dim=1)
        return x

    def _load_dicom(self, path):
        dcm = pydicom.dcmread(path)
        pixels = dcm.pixel_array
        if pixels.ndim == 3:
            pixels = np.repeat(pixels[..., None], 3, axis=3)

        pixels = self.mask_outside_ultrasound(pixels)
        x = np.zeros((len(pixels), 224,224,3), dtype=np.float16)
        for i in range(len(x)):
            x[i] = self.crop_and_scale(pixels[i])

        x = torch.from_numpy(x).permute(3,0,1,2)
        x = x[:, ::self.frame_stride][:, :self.frames_to_take]
        x.sub_(self.mean).div_(self.std)

        if x.shape[1] < self.frames_to_take:
            pad = torch.zeros(3, self.frames_to_take - x.shape[1], 224,224, dtype=torch.half)
            x = torch.cat([x, pad], dim=1)
        return x

    # ====================== 视觉特征 ======================
    @torch.no_grad()
    def get_visual_embeds(self, video_tensor):
        feat = self.echo_encoder(video_tensor)
        feat = F.normalize(feat, dim=-1)
        first_frame = video_tensor[:, :, 0, :, :]
        view_logits = self.view_classifier(first_frame)
        view_onehot = F.one_hot(torch.argmax(view_logits, dim=1), 11).float()
        concat = torch.cat([feat, view_onehot], dim=1).half()
        return self.visual_projection(concat)

    # ====================== 生成报告 ======================
    @torch.no_grad()
    def generate_report(self, video_tensor, prompt="Please generate a complete echocardiogram report."):
        visual = self.get_visual_embeds(video_tensor)

        prompt_full = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        tokens = self.tokenizer.encode(
            prompt_full, add_special_tokens=True, return_tensors="pt"
        ).to(self.device)

        text_embeds = self.medgemma.get_input_embeddings()(tokens)
        embeds = torch.cat([visual, text_embeds], dim=1)
        mask = torch.ones(1, embeds.shape[1], device=self.device)

        outputs = self.medgemma.generate(
            inputs_embeds=embeds,
            attention_mask=mask,
            max_new_tokens=1024,
            temperature=0.1,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("model\n")[-1].strip()

    # ====================== 工具 ======================
    @staticmethod
    def mask_outside_ultrasound(pixels):
        try:
            vid = pixels.copy()
            mask = np.ones(vid.shape[1:3], dtype=np.uint8) * 255
            return vid * mask[None,...,None]
        except:
            return pixels

    @staticmethod
    def crop_and_scale(img, res=(224,224), zoom=0.1):
        h, w = img.shape[:2]
        r_in, r_out = w / h, res[0] / res[1]
        if r_in > r_out:
            p = int((w - r_out * h) / 2)
            img = img[:, p:-p]
        elif r_in < r_out:
            p = int((h - w / r_out) / 2)
            img = img[p:-p]
        if zoom > 0:
            px, py = round(img.shape[1] * zoom), round(img.shape[0] * zoom)
            img = img[py:-py, px:-px]
        return cv2.resize(img, res, interpolation=cv2.INTER_CUBIC)

# ====================== 框架主逻辑 ======================
def run_pipeline(video_dir, save_path):
    print("Loading EchoGemma...")
    model = EchoGemmaVQA()

    print(f"Processing videos from: {video_dir}")
    video_tensor = model.process_video_folder(video_dir)

    print("Generating report...")
    report = model.generate_report(video_tensor)

    os.makedirs(Path(save_path).parent, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report)

    print("✅ Report saved successfully!")
    print("\n=== Generated Report ===")
    print(report)
    return report

# ====================== Click 命令 ======================
@click.command("visual_question_answering_echo")
@click.option("--video_dir", type=str, required=True, help="Video/DICOM folder path")
@click.option("--save_path", type=str, required=True, help="Path to save .txt report")
def run(video_dir, save_path):
    run_pipeline(video_dir, save_path)

# ====================== 引擎注册 ======================
def register():
    return {
        "name": "visual_question_answering_echo",
        "entry": run,
        "description": "EchoGemma 超声视频智能报告生成"
    }

if __name__ == "__main__":
    run()