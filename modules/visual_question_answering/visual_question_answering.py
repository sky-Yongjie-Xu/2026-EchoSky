# -*- coding: utf-8 -*-
import os
import torch
import cv2
import click
import numpy as np
from pathlib import Path
from PIL import Image
import pydicom
from transformers import AutoProcessor, AutoModelForImageTextToText

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ====================== 全局加速配置 ======================
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# ====================== 模型只加载一次 缓存 ======================
_MODEL_CACHE = {}

def load_model_once(model_id):
    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id]
    
    print("⚡ 加载 MedGemma (仅第一次加载)...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    _MODEL_CACHE[model_id] = (model, processor)
    return model, processor

# ====================== DCM 读取 ======================
def load_dcm_fast(dcm_path):
    ds = pydicom.dcmread(dcm_path)
    img = ds.pixel_array

    while img.ndim > 3:
        img = img.squeeze()
    if img.ndim == 3 and img.shape[0] in (1,3):
        img = img.transpose(1,2,0)
    if img.ndim == 2:
        img = np.stack([img,img,img], axis=-1)

    img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
    return Image.fromarray(img.astype(np.uint8))

# ====================== 视频读取 ======================
def load_video_fast(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total//2)
    ret, frame = cap.read()
    cap.release()
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# ====================== 自动加载媒体 ======================
def load_media(path):
    if path.lower().endswith(".dcm"):
        return load_dcm_fast(path)
    else:
        return load_video_fast(path)

# ====================== 高速对话（已修复报错） ======================
def fast_chat(model, processor, image, chat_history, user_input):
    prompt = f"请用中文专业回答：{user_input}"

    chat_history.append({
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]
    })

    # 🔥 修复：必须返回 dict，不能只返回 tensor
    inputs = processor.apply_chat_template(
        chat_history,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to("cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    response = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    chat_history.append({
        "role": "assistant", 
        "content": [{"type": "text", "text": response}]
    })
    return response

# ====================== 交互主程序 ======================
def interactive_chat(media_path, model_id):
    model, processor = load_model_once(model_id)
    image = load_media(media_path)
    chat_history = []

    print("\n" + "="*50)
    print("🏥 MedGemma 极速中文版（1~3秒出答案）")
    print("="*50)

    while True:
        user = input("\n你：").strip()
        if user in ["quit", "exit"]: 
            break
        if user == "clear": 
            chat_history = []
            print("✅ 对话已清空")
            continue

        print("🤖 生成中...")
        ans = fast_chat(model, processor, image, chat_history, user)
        print(f"\n助手：\n{ans}\n")

# ====================== 命令行接口 ======================
@click.command("visual_question_answering")
@click.option("--media", type=str, required=True, help="dcm或视频路径")
@click.option("--model-id", default="modules/report_generation/medgemma-1.5-4b-it")
def run(media, model_id):
    interactive_chat(media, model_id)

# ====================== 引擎注册 ======================
def register():
    return {
        "name": "visual_question_answering",
        "entry": run,
        "description": "MedGemma 极速中文超声问答"
    }

if __name__ == "__main__":
    run()