# -*- coding: utf-8 -*-
import os
import click
import sys
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from modules.report_generation.echogemma import EchoGemma


# ==============================================
# 核心推理 pipeline
# ==============================================
def run_pipeline(
    dicom_dir: str,
    save_path: str
):
    print("🔹 Loading EchoGemma model...")
    eg = EchoGemma()

    print(f"🔹 Processing DICOM study: {dicom_dir}")
    stack_of_videos = eg.process_dicoms(dicom_dir)

    print("🔹 Generating report...")
    report = eg.generate(stack_of_videos)

    # 保存报告到文件
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report)

    print("✅ Report generated successfully!")
    print("=" * 50)
    print(report)
    print("=" * 50)

    return report


# ==============================================
# 命令行接口（和你框架统一）
# ==============================================
@click.command("report_generation_gemma")
@click.option("--dicom_dir", type=str, required=True, help="DICOM 文件夹路径")
@click.option("--save_path", type=str, required=True, help="报告保存路径 .txt")
def run(dicom_dir, save_path):
    run_pipeline(
        dicom_dir=dicom_dir,
        save_path=save_path
    )


# ==============================================
# 引擎注册（和你项目完全兼容）
# ==============================================
def register():
    return {
        "name": "report_generation_gemma",
        "entry": run,
        "description": "EchoGemma 超声智能报告生成"
    }


if __name__ == "__main__":
    run()