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

# ====================== 显存&速度优化 7G专用 ======================
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ====================== 模型全局缓存（只加载一次） ======================
_MODEL_CACHE = {}

def load_model_once(model_id):
    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id]

    print("⚡ 正在加载 MedGemma 医学多模态模型...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    _MODEL_CACHE[model_id] = (model, processor)
    print("✅ 模型加载完成")
    return model, processor

# ====================== DICOM 高清读取 ======================
def load_dcm_fast(dcm_path):
    ds = pydicom.dcmread(dcm_path)
    img = ds.pixel_array

    while img.ndim > 3:
        img = img.squeeze()
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = img.transpose(1, 2, 0)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
    return Image.fromarray(img.astype(np.uint8)).convert("RGB")

# ====================== 视频单帧抽取 ======================
def load_video_fast(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ret, frame = cap.read()
    cap.release()
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# ====================== 媒体自动适配 ======================
def load_media(path):
    if path.lower().endswith(".dcm"):
        return load_dcm_fast(path)
    else:
        return load_video_fast(path)

# ====================== 核心对话｜终极稳定版 ======================
def fast_chat(model, processor, image, chat_history, user_input):
    system_prompt = (
        "你是专业心脏超声诊断医师。请使用标准简体中文回答，禁止英文，术语规范，客观严谨。\n"
        "问题："
    )
    full_text = system_prompt + user_input

    # 限制历史长度，防止超长
    chat_history = chat_history[-4:]
    
    new_history = []
    for turn in chat_history:
        if turn["role"] == "user":
            new_history.append({
                "role": "user",
                "content": [{"type": "text", "text": turn["content"]}]
            })
        elif turn["role"] == "assistant":
            new_history.append({
                "role": "assistant",
                "content": [{"type": "text", "text": turn["content"]}]
            })

    new_history.append({
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": full_text}
        ]
    })

    # 安全配置，永不崩溃
    inputs = processor.apply_chat_template(
        new_history,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        truncation=False,
        max_length=4096,
    ).to("cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=768,
            do_sample=False,
            use_cache=True,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    response = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": response})
    return response

# ====================== 交互入口 ======================
def interactive_chat(media_path, model_id):
    model, processor = load_model_once(model_id)
    image = load_media(media_path)
    chat_history = []

    print("\n" + "="*55)
    print("🏥 心脏超声智能问答系统 | 全程中文 | 稳定不崩溃")
    print("     quit 退出 ｜ clear 清空对话")
    print("="*55)

    while True:
        user = input("\n你：").strip()
        if user.lower() in ["quit", "exit"]:
            print("👋 对话结束")
            break
        if user.lower() == "clear":
            chat_history = []
            print("✅ 对话历史已清空")
            continue
        if not user:
            continue

        print("🤖 分析中...")
        ans = fast_chat(model, processor, image, chat_history, user)
        print(f"\n助手：\n{ans}")

# ====================== Click 命令行 ======================
@click.command("visual_question_answering_medgemma")
@click.option("--media", type=str, required=True, help="DCM文件 / 超声视频路径")
@click.option("--model-id", type=str, default="modules/report_generation/medgemma-1.5-4b-it")
def run(media, model_id):
    interactive_chat(media, model_id)

# ====================== 引擎注册 ======================
def register():
    return {
        "name": "visual_question_answering_medgemma",
        "entry": run,
        "description": "MedGemma 心脏超声多轮中文问答（稳定版）"
    }

if __name__ == "__main__":
    run()