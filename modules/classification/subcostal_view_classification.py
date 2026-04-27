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
from modules.classification.utils import EchoDataset, sigmoid

# ==============================================
# 模型类
# ==============================================
class SubcostalViewClassifier:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        BASE = os.path.dirname(os.path.abspath(__file__))
        self.weights_path = os.path.join(BASE, "weights/subcostal_view_classifier_model.pt")

        print(f"[LOAD] View Classifier: {self.weights_path}")

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

        for batch in tqdm(test_dl, desc="View Classifier 推理中"):
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
    model = SubcostalViewClassifier()
    manifest = pd.read_csv(manifest_path)
    manifest["split"] = "test"
    if 'file_uid' in manifest.columns:
        manifest = manifest.rename(columns={'file_uid': 'filename'})
    if 'filename' in manifest.columns and manifest['filename'].str.contains('.avi').all() == False:
        manifest['filename'] = manifest['filename'].apply(lambda x: x + '.avi')
    manifest.to_csv(manifest_path, index=False)

    df_preds = model.predict(dataset, manifest_path)
    out = manifest.merge(df_preds, on="filename", how="inner").drop_duplicates('filename')
    out.preds = out.preds.apply(sigmoid)

    out_above = out[out.preds > 0.8414]

    if not output_csv:
        output_csv = os.path.join(os.getcwd(), "view_classification_predictions_above_threshold.csv")

    out_above.to_csv(output_csv, index=False)
    print(f"[✅] 输出：{output_csv}")
    return output_csv

# ==============================================
# Click 命令
# ==============================================
@click.command("subcostal_view_classification")
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
        "name": "subcostal_view_classification",
        "entry": run,
        "description": "Subcostal 视图分类（高质量筛选 Step1）"
    }

if __name__ == "__main__":
    run()