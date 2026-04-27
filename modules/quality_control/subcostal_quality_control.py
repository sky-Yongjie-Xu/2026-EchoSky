# -*- coding: utf-8 -*-
import os
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import click
from torch.utils.data import DataLoader
from torchvision.models.video import r2plus1d_18

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from modules.quality_control.utils import EchoDataset, sigmoid

# ==============================================
# 模型类
# ==============================================
class SubcostalQualityControl:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        BASE = os.path.dirname(os.path.abspath(__file__))
        self.weights_path = os.path.join(BASE, "weights/quality_control_model.pt")

        print(f"[LOAD] Quality Control: {self.weights_path}")

        weights = torch.load(self.weights_path, map_location=self.device, weights_only=True)
        new_state_dict = {}
        for k, v in weights.items():
            new_key = k[2:] if k.startswith('m.') else k
            new_state_dict[new_key] = v

        self.model = r2plus1d_18(num_classes=1)
        self.model.load_state_dict(new_state_dict, strict=True)
        self.model = self.model.to(self.device).eval()

    def predict(self, data_path, manifest_path):
        test_ds = EchoDataset(split="test", data_path=data_path, manifest_path=manifest_path)
        test_dl = DataLoader(test_ds, num_workers=8, batch_size=10, drop_last=False, shuffle=False)

        filenames = []
        predictions = []

        for batch in tqdm(test_dl, desc="Quality Control 推理中"):
            with torch.no_grad():
                preds = self.model(batch["primary_input"].to(self.device))
            filenames.extend(batch["filename"])
            predictions.extend(preds.detach().cpu().squeeze(dim=1))

        df_preds = pd.DataFrame({'filename': filenames, 'preds': predictions})
        return df_preds

# ==============================================
# 执行 pipeline
# ==============================================
def run_pipeline(dataset, manifest_path, output_csv=None):
    model = SubcostalQualityControl()
    df_preds = model.predict(dataset, manifest_path)

    manifest = pd.read_csv(manifest_path)
    if 'preds' in manifest.columns:
        manifest = manifest.drop(columns=['preds'])

    out = manifest.merge(df_preds, on="filename", how="inner").drop_duplicates('filename')
    out.preds = out.preds.apply(sigmoid)
    out_above = out[out.preds > 0.925]

    if not output_csv:
        output_csv = os.path.join(os.getcwd(), "modules/quality_control/outputs/subcostal_quality_control_predictions_above_threshold.csv")

    out_above.to_csv(output_csv, index=False)
    print(f"[✅] 输出：{output_csv}")
    return output_csv

# ==============================================
# Click 命令
# ==============================================
@click.command("subcostal_quality_control")
@click.option("--dataset", "-d", required=True, type=str)
@click.option("--manifest_path", "-m", required=True, type=str)
@click.option("--output_csv", "-o", default=None, type=str)
def run(dataset, manifest_path, output_csv):
    run_pipeline(dataset, manifest_path, output_csv)

# ==============================================
# 注册引擎
# ==============================================
def register():
    return {
        "name": "subcostal_quality_control",
        "entry": run,
        "description": "Subcostal 质量控制（高质量筛选 Step2）"
    }

if __name__ == "__main__":
    run()