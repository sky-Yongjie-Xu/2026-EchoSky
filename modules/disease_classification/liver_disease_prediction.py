# -*- coding: utf-8 -*-
import os
import torch
import pandas as pd
from tqdm import tqdm
import click
from torch.utils.data import DataLoader
from torchvision.models import densenet121

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from modules.disease_classification.utils import EchoDataset, sigmoid

# import pytorch_lightning
# from lightning_utilities.core.imports import compare_version

# ==============================================
# 模型预测类
# ==============================================
class LiverPredictor:
    def __init__(self, label):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.label = label
        BASE = os.path.dirname(os.path.abspath(__file__))

        if label == "cirrhosis":
            self.weights = os.path.join(BASE, "weights/pretrained_model_weight_cirrhosis.pt")
        elif label == "SLD":
            self.weights = os.path.join(BASE, "weights/pretrained_model_weight_fattyliver.pt")

        weights = torch.load(self.weights, map_location=self.device)
        new_state_dict = {k[2:] if k.startswith("m.") else k: v for k, v in weights.items()}

        self.model = densenet121(pretrained=False)
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = torch.nn.Linear(num_ftrs, 1)
        self.model.load_state_dict(new_state_dict, strict=True)
        self.model = self.model.to(self.device).eval()

    def predict(self, data_path, manifest_path):
        test_ds = EchoDataset(
            split="test",
            data_path=data_path,
            manifest_path=manifest_path,
            n_frames=1,
            resize_res=(480, 640),
            random_start=True
        )

        test_dl = DataLoader(test_ds, num_workers=8, batch_size=10, shuffle=False)

        filenames = []
        predictions = []

        with torch.no_grad():
            for batch in tqdm(test_dl):
                preds = self.model(batch["primary_input"].to(self.device))
                filenames.extend(batch["filename"])
                predictions.extend(preds.detach().cpu().squeeze(dim=1))

        return pd.DataFrame({"filename": filenames, "preds": predictions})

# ==============================================
# 核心 pipeline
# ==============================================
def run_pipeline(dataset, manifest_path, label):
    model = LiverPredictor(label)
    df_preds = model.predict(dataset, manifest_path)

    manifest = pd.read_csv(manifest_path)
    out = manifest.merge(df_preds, on="filename", how="inner").drop_duplicates("filename")
    out["preds"] = out["preds"].apply(sigmoid)

    output_csv = f"modules/disease_classification/outputs/disease_detection_{label}.csv"
    out.to_csv(output_csv, index=False)
    print(f"✅ 输出结果：{output_csv}")

# ==============================================
# Click 命令（框架格式）
# ==============================================
@click.command("liver_disease_prediction")
@click.option("--dataset", "-d", required=True, type=str)
@click.option("--manifest_path", "-m", required=True, type=str)
@click.option("--label", "-l", required=True, type=str, help="疾病标签：cirrhosis 或 SLD")
def run(dataset, manifest_path, label):
    run_pipeline(dataset, manifest_path, label)

# ==============================================
# 引擎注册
# ==============================================
def register():
    return {
        "name": "liver_disease_prediction",
        "entry": run,
        "description": "肝脏疾病预测（cirrhosis / SLD）"
    }

if __name__ == "__main__":
    run()