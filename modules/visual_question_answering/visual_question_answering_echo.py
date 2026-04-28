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
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType

# 自动加入项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ====================== 显存优化 ======================
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
os.environ["TOKENIZER_PARALLELISM"] = "false"

# ====================== 量化配置 ======================
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_use_double_quant=True,
    bnb_8bit_quant_type="int8"
)
model_dtype = torch.bfloat16

# 全局模型缓存
_MODEL_CACHE = {}

# ====================== EchoGemma 模型 ======================
class EchoGemmaVQA(nn.Module):
    def __init__(self, model_id="modules/report_generation/medgemma-1.5-4b-it", 
                 weight_path="modules/report_generation/weights/echogemma.pt",
                 emb_dim=512, device=torch.device('cuda')):
        super().__init__()
        self.device = device
        self.model_id = model_id
        self.weight_path = weight_path

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # 大模型
        self.medgemma = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
            torch_dtype=model_dtype
        )

        # LoRA
        lora_config = LoraConfig(
            r=16, lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM
        )
        self.medgemma = get_peft_model(self.medgemma, lora_config)
        self.medgemma.eval()

        # 视频编码器
        self.echo_encoder = torchvision.models.video.mvit_v2_s()
        self.echo_encoder.head[-1] = nn.Linear(self.echo_encoder.head[-1].in_features, emb_dim)

        # 视图分类
        self.view_classifier = torchvision.models.convnext_base()
        self.view_classifier.classifier[-1] = nn.Linear(self.view_classifier.classifier[-1].in_features, 11)

        # 投影层
        self.visual_projection = nn.Linear(emb_dim + 11, 2560, dtype=model_dtype)

        # 视频预处理
        self.frames_to_take = 16
        self.frame_stride = 2
        self.video_size = 224
        self.mean = torch.tensor([29.1106, 28.0768, 29.0964]).reshape(3,1,1,1).to(self.device)
        self.std = torch.tensor([47.9892, 46.4570, 47.2008]).reshape(3,1,1,1).to(self.device)

        # 加载权重
        self._load_weights()

        # 推理模式
        self.to(self.device).to(model_dtype).eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def _load_weights(self):
        if not Path(self.weight_path).exists():
            print(f"⚠️  未找到权重：{self.weight_path}")
            return
        try:
            checkpoint = torch.load(self.weight_path, map_location="cpu")
            self.load_state_dict(checkpoint, strict=False)
            print("✅ 权重加载完成")
        except:
            print("❌ 权重加载失败")

    # ====================== 媒体加载 ======================
    def load_media_auto(self, path):
        return self._load_single_media(path)

    def _load_single_media(self, file_path):
        if file_path.endswith(".dcm"):
            video = self._load_dicom(file_path)
        else:
            video = self._load_video(file_path)
        return video.unsqueeze(0).to(self.device)

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while len(frames) < self.frames_to_take:
            ret, f = cap.read()
            if not ret: break
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            f = self.crop_and_scale(f)
            frames.append(f)
        cap.release()

        x = torch.from_numpy(np.array(frames, np.float32)).permute(3,0,1,2)
        x = x[:, ::self.frame_stride][:, :self.frames_to_take]
        x = (x - self.mean) / self.std
        if x.shape[1] < self.frames_to_take:
            pad = torch.zeros(3, self.frames_to_take-x.shape[1], 224,224, dtype=model_dtype)
            x = torch.cat([x, pad], dim=1)
        return x.to(model_dtype)

    def _load_dicom(self, path):
        pixels = pydicom.dcmread(path).pixel_array
        if pixels.ndim == 3:
            pixels = np.repeat(pixels[..., None], 3, axis=-1)

        x = np.stack([self.crop_and_scale(f) for f in pixels])
        x = torch.from_numpy(x).permute(3,0,1,2)
        x = x[:, ::self.frame_stride][:, :self.frames_to_take]
        x = (x - self.mean) / self.std
        if x.shape[1] < self.frames_to_take:
            pad = torch.zeros(3, self.frames_to_take-x.shape[1],224,224,dtype=model_dtype)
            x = torch.cat([x,pad],dim=1)
        return x.to(model_dtype)

    # ====================== 视觉特征 ======================
    @torch.no_grad()
    def get_visual_embeds(self, video_tensor):
        video_tensor = video_tensor.to(self.device, model_dtype)
        feat = F.normalize(self.echo_encoder(video_tensor), dim=-1)
        view = self.view_classifier(video_tensor[:,:,0])
        view = F.one_hot(view.argmax(1),11).to(model_dtype)
        return self.visual_projection(torch.cat([feat,view],1)).unsqueeze(1)

    # ====================== 核心问答 ======================
    @torch.no_grad()
    def chat(self, video_tensor, user_input, chat_history):
        system = "你是专业心脏超声医师，用中文严谨回答，禁止英文，术语规范。\n问题："
        prompt = system + user_input
        chat_history = chat_history[-4:]

        hist = []
        for turn in chat_history:
            if turn["role"] == "user":
                hist.append(f"<start_of_turn>user\n{turn['content']}<end_of_turn>")
            else:
                hist.append(f"<start_of_turn>model\n{turn['content']}<end_of_turn>")

        full = f"{''.join(hist)}<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        visual = self.get_visual_embeds(video_tensor)
        tokens = self.tokenizer.encode(full, return_tensors="pt").to(self.device)
        text_embeds = self.medgemma.get_input_embeddings()(tokens)
        embeds = torch.cat([visual, text_embeds], dim=1)
        mask = torch.ones(1, embeds.shape[1], device=self.device)

        out = self.medgemma.generate(
            inputs_embeds=embeds, attention_mask=mask,
            max_new_tokens=768, temperature=0.2, do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id
        )

        ans = self.tokenizer.decode(out[0], skip_special_tokens=True).split("model\n")[-1].strip()
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": ans})
        return ans

    # ====================== 工具 ======================
    @staticmethod
    def crop_and_scale(img, res=(224,224), zoom=0.1):
        h, w = img.shape[:2]
        r = w/h
        tr = res[0]/res[1]
        if r > tr:
            p = int((w-tr*h)/2)
            img = img[:,p:-p]
        else:
            p = int((h-w/tr)/2)
            img = img[p:-p]
        if zoom>0:
            px,py=int(img.shape[1]*zoom),int(img.shape[0]*zoom)
            img=img[py:-py,px:-px] if py>0 else img
        return cv2.resize(img,res,interpolation=cv2.INTER_CUBIC)

# ====================== 模型单例 ======================
def get_model():
    if "echogemma" not in _MODEL_CACHE:
        _MODEL_CACHE["echogemma"] = EchoGemmaVQA()
    return _MODEL_CACHE["echogemma"]

# ====================== 交互对话 ======================
def interactive_chat(media_path):
    model = get_model()
    video_tensor = model.load_media_auto(media_path)
    chat_history = []

    print("\n" + "="*50)
    print("🏥 心脏超声智能问答 | 全程中文")
    print("     quit 退出 ｜ clear 清空对话")
    print("="*50)

    while True:
        user = input("\n你：").strip()
        if user in ["quit", "exit"]:
            print("👋 对话结束")
            break
        if user == "clear":
            chat_history = []
            print("✅ 已清空")
            continue
        if not user:
            continue
        print("🤖 分析中...")
        ans = model.chat(video_tensor, user, chat_history)
        print(f"\n医师：\n{ans}")

# ====================== 命令行（仅保留问答） ======================
@click.command("visual_question_answering_echo")
@click.option("--media", type=str, required=True, help="视频/DICOM文件路径")
def run(media):
    interactive_chat(media)

# ====================== 引擎注册 ======================
def register():
    return {
        "name": "visual_question_answering_echo",
        "entry": run,
        "description": "EchoGemma 心脏超声智能问答（仅对话）"
    }

if __name__ == "__main__":
    run()