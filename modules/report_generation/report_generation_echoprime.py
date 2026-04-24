# -*- coding: utf-8 -*-
# Standard library imports
import os
import math
import glob
import json
import pickle
import random

# Third-party library imports
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import pydicom
import sklearn
import sklearn.metrics
import transformers
import click

# Local module imports
import utils

# ==============================================
# 直接把 EchoPrime 类写在本文件内（不外部导入）
# ==============================================
class EchoPrime:
    def __init__(self, device=None, lang='en'):
        utils.initialize_language(lang)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 视频编码器
        checkpoint = torch.load("modules/report_generation/weights/echo_prime_encoder.pt", map_location=device)
        echo_encoder = torchvision.models.video.mvit_v2_s()
        echo_encoder.head[-1] = torch.nn.Linear(echo_encoder.head[-1].in_features, 512)
        echo_encoder.load_state_dict(checkpoint)
        echo_encoder.eval().to(device)
        for param in echo_encoder.parameters():
            param.requires_grad = False

        # 切面分类器
        vc_state_dict = torch.load("modules/report_generation/weights/view_classifier.pt")
        view_classifier = torchvision.models.convnext_base()
        view_classifier.classifier[-1] = torch.nn.Linear(view_classifier.classifier[-1].in_features, 11)
        view_classifier.load_state_dict(vc_state_dict)
        view_classifier.eval().to(device)
        for param in view_classifier.parameters():
            param.requires_grad = False

        self.echo_encoder = echo_encoder
        self.view_classifier = view_classifier
        self.frames_to_take = 32
        self.frame_stride = 2
        self.video_size = 224
        self.mean = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(3, 1, 1, 1)
        self.std = torch.tensor([47.989223, 46.456997, 47.20083]).reshape(3, 1, 1, 1)
        self.device = device
        self.lang = lang

        # 权重与报告库
        self.MIL_weights = pd.read_csv("assets/MIL_weights.csv")
        self.non_empty_sections = self.MIL_weights['Section']
        self.section_weights = self.MIL_weights.iloc[:,1:].to_numpy()

        self.candidate_studies = list(pd.read_csv("modules/report_generation/candidates_data/candidate_studies.csv")['Study'])
        ce1 = torch.load("modules/report_generation/candidates_data/candidate_embeddings_p1.pt")
        ce2 = torch.load("modules/report_generation/candidates_data/candidate_embeddings_p2.pt")
        self.candidate_embeddings = torch.cat((ce1, ce2), dim=0)

        cr = pd.read_pickle("modules/report_generation/candidates_data/candidate_reports.pkl")
        self.candidate_reports = [utils.phrase_decode(p) for p in tqdm(cr)]
        self.candidate_labels = pd.read_pickle("modules/report_generation/candidates_data/candidate_labels.pkl")
        self.section_to_phenotypes = pd.read_pickle("assets/section_to_phenotypes.pkl")

    def process_dicoms(self,INPUT):
        """
        Reads DICOM video data from the specified folder and returns a tensor 
        formatted for input into the EchoPrime model.

        Args:
            INPUT (str): Path to the folder containing DICOM files.

        Returns:
            stack_of_videos (torch.Tensor): A float tensor of shape  (N, 3, 16, 224, 224)
                                            representing the video data where N is the number of videos,
                                            ready to be fed into EchoPrime.
        """

        dicom_paths = glob.glob(f'{INPUT}/**/*.dcm',recursive=True)
        stack_of_videos=[]
        for idx, dicom_path in tqdm(enumerate(dicom_paths),total=len(dicom_paths)):
            try:
                # simple dicom_processing
                dcm=pydicom.dcmread(dicom_path)
                pixels = dcm.pixel_array
                
                # exclude images like (600,800) or (600,800,3)
                if pixels.ndim < 3 or pixels.shape[2]==3:
                    continue 
                    
                # if single channel repeat to 3 channels    
                if pixels.ndim==3:
                    
                    pixels = np.repeat(pixels[..., None], 3, axis=3)
                
                # mask everything outside ultrasound region
                pixels=utils.mask_outside_ultrasound(dcm.pixel_array)
                
                #model specific preprocessing
                x = np.zeros((len(pixels),224,224,3))
                for i in range(len(x)):
                    x[i] = utils.crop_and_scale(pixels[i])
                
                x = torch.as_tensor(x, dtype=torch.float).permute([3,0,1,2])
                # normalize
                x.sub_(self.mean).div_(self.std)
            
                ## if not enough frames add padding
                if x.shape[1] < self.frames_to_take:
                    padding = torch.zeros(
                    (
                        3,
                        self.frames_to_take - x.shape[1],
                        self.video_size,
                        self.video_size,
                    ),
                    dtype=torch.float,
                    )
                    x = torch.cat((x, padding), dim=1)
                    
                start=0
                stack_of_videos.append(x[:, start : ( start + self.frames_to_take) : self.frame_stride, : , : ])
                
            except Exception as e:
                print("corrupt file")
                print(str(e))

        stack_of_videos=torch.stack(stack_of_videos)
        
        return stack_of_videos

    def embed_videos(self, vids):
        bins = math.ceil(len(vids)/50)
        feats = []
        with torch.no_grad():
            for i in range(bins):
                s = i*50
                e = min((i+1)*50, len(vids))
                feats.append(self.echo_encoder(vids[s:e].to(self.device)))
        return torch.cat(feats, dim=0)

    def get_views(self, vids, visualize=False, return_view_list=False):
        frames = vids[:,:,0].to(self.device)
        with torch.no_grad():
            logits = self.view_classifier(frames)
        preds = torch.argmax(logits, dim=1)
        view_list = [utils.COARSE_VIEWS[p] for p in preds]
        if return_view_list:
            return view_list
        return F.one_hot(preds, 11).float().to(self.device)

    @torch.no_grad()
    def encode_study(self, vids, visualize=False):
        feats = self.embed_videos(vids)
        views = self.get_views(vids, visualize)
        return torch.cat([feats, views], dim=1)

    def translate_sections(self, rep):
        if self.lang != 'zh':
            return rep
        trans = {
            "Left Ventricle": "左心室",
            "Resting Segmental Wall Motion Analysis": "静息节段性室壁运动分析",
            "Right Ventricle": "右心室", "Left Atrium": "左心房", "Right Atrium": "右心房",
            "Atrial Septum": "房间隔", "Mitral Valve": "二尖瓣", "Aortic Valve": "主动脉瓣",
            "Tricuspid Valve": "三尖瓣", "Pulmonic Valve": "肺动脉瓣", "Pericardium": "心包",
            "Aorta": "主动脉", "IVC": "下腔静脉", "Pulmonary Artery": "肺动脉",
            "Pulmonary Veins": "肺静脉", "Postoperative Findings": "术后表现"
        }
        for k, v in trans.items():
            rep = rep.replace(k, v)
        return rep

    def generate_report(self, emb):
        emb = emb.cpu()
        out = ""
        for i, sec in enumerate(self.non_empty_sections):
            ws = [self.section_weights[i][torch.where(v==1)[0]] for v in emb[:,512:]]
            wemb = emb[:,:512] * torch.tensor(ws).unsqueeze(1)
            wemb = wemb.mean(dim=0)
            wemb = F.normalize(wemb, dim=0)
            sim = wemb.float() @ self.candidate_embeddings.T
            txt = "Section not found."
            while txt == "Section not found.":
                mid = torch.argmax(sim)
                cand = self.candidate_reports[mid]
                txt = utils.extract_section(cand, sec)
                sim[mid] = -float('inf')
            out += txt + "\n\n"
        return self.translate_sections(out)

    def predict_metrics(self, emb, k=50):
        n_sec = len(self.non_empty_sections)
        sec_emb = torch.zeros(n_sec, 512)
        emb = emb.cpu()
        for i, sec in enumerate(self.non_empty_sections):
            ws = [self.section_weights[i][torch.where(v==1)[0]] for v in emb[:,512:]]
            wemb = emb[:,:512] * torch.tensor(ws).unsqueeze(1)
            sec_emb[i] = wemb.sum(dim=0)
        sec_emb = F.normalize(sec_emb)
        sim = sec_emb @ self.candidate_embeddings.T
        topk = torch.topk(sim, k=k, dim=1).indices
        preds = {}
        for i, sec in enumerate(self.section_to_phenotypes):
            for ph in self.section_to_phenotypes[sec]:
                vals = []
                for mid in topk[i]:
                    st = self.candidate_studies[mid]
                    if st in self.candidate_labels[ph]:
                        vals.append(self.candidate_labels[ph][st])
                preds[ph] = np.nanmean(vals) if vals else np.nan
        return preds

# ==============================================
# 命令行入口（和你所有插件格式一致）
# ==============================================
@click.command("report_generation_echoprime")
@click.option("--dataset_dir", "-i", required=True, help="DICOM 路径")
@click.option("--lang", "-l", default="zh", help="语言 en/zh")
@click.option("--output", "-o", default="modules/report_generation/output/report.txt", help="保存路径")
@click.option("--visualize", is_flag=True, help="可视化切面")
def run(dataset_dir, lang, output, visualize):
    print("[+] 加载模型...")
    ep = EchoPrime(lang=lang)

    print("[+] 处理视频...")
    videos = ep.process_dicoms(dataset_dir)
    emb = ep.encode_study(videos, visualize=visualize)

    print("[+] 生成报告...\n")
    report = ep.generate_report(emb)

    print("="*50)
    print("📄 生成的超声报告")
    print("="*50)
    print(report)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n✅ 报告已保存: {output}")

# ==============================================
# 引擎注册
# ==============================================
def register():
    return {
        "name": "report_generation_echoprime",
        "entry": run,
        "description": "EchoPrime 全自动超声报告生成"
    }

if __name__ == "__main__":
    run()