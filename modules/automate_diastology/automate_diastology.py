# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import numpy as np
import pydicom
from pathlib import Path
import tqdm
import sys
import os
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import warnings
import click

warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from modules.automate_diastology.utils import ase_guidelines, dicom_utils, model_utils, lav_mask

# ====================== 全局配置 ======================
weights_dir = Path.cwd().parent / 'EchoSky/modules'

diastology_views = [
    'BMode_A4C', 'BMode_A4C_LV', 'BMode_A2C',
    'DOPPLER_A4C_MV_PW',
    'TDI_MV_Medial e',
    'TDI_MV_Lateral e',
    'DOPPLER_A4C_TV_CW', 'DOPPLER_PSAX_Great_vessel_level_TV_CW'
]

doppler_names = {
    'MEDEVEL': ['TDI_MV_Medial e'],
    'LATEVEL': ['TDI_MV_Lateral e'],
    'TRVMAX': ['DOPPLER_A4C_TV_CW', 'DOPPLER_PSAX_Great_vessel_level_TV_CW']
}

scale = 800 / 112 * 600 / 112 * 600 / 112

# ====================== 核心业务逻辑 ======================
def run_diastology_pipeline(path, guideline_year, quality_threshold=0.0, to_save=True, save_path=None):
    path = Path(path)
    ase_year = guideline_year

    if save_path is None:
        save_path = path / 'diastology_results'
    save_path = Path(save_path)

    if not save_path.exists():
        os.mkdir(save_path)

    if not path.exists():
        print(f'❌ 路径不存在: {path}')
        return

    # ====================== DICOM 预处理 ======================
    print("\n🔹 正在读取 DICOM 文件...")
    files = [f for f in path.iterdir() if f.is_file()]
    image_dataset = {}
    video_dataset = {}
    bsa = 0.0

    for f in files:
        dcm_path = Path(path / f)
        if bsa == 0:
            bsa = dicom_utils.get_bsa(dcm_path)
        pixels = dicom_utils.change_dicom_color(dcm_path)

        if len(pixels.shape) == 4:
            if pixels.shape[0] >= 32:
                x, h0, w0 = dicom_utils.convert_video_dicom(pixels)
                x_first_frame = dicom_utils.pull_first_frame(x)
                image_dataset[f] = x_first_frame
                video_dataset[f] = x
            else:
                print(f'⚠️ 帧数不足: {dcm_path.name}, {pixels.shape[0]}')
        else:
            x = dicom_utils.convert_image_dicom(pixels)
            if x is None:
                print(f'⚠️ 不支持的文件: {dcm_path.name}')
                continue
            image_dataset[f] = x

    # ====================== 视图分类 ======================
    print(f"\n🔹 正在分类视图 ({len(image_dataset)} 个文件)")
    view_input = torch.stack(list(image_dataset.values()))
    filenames = list(image_dataset.keys())
    view_model = model_utils.load_view_classifier()
    predicted_view = model_utils.view_inference(view_input, view_model, filenames)
    view_df = pd.DataFrame({
        'filename': list(predicted_view.keys()),
        'predicted_view': list(predicted_view.values())
    })

    if to_save:
        view_df.to_csv(save_path / 'predicted_views.csv', index=None)

    to_remove = view_df[~view_df.predicted_view.isin(diastology_views)].filename
    for f in to_remove:
        image_dataset.pop(f, None)
        video_dataset.pop(f, None)
    view_df = view_df[view_df.predicted_view.isin(diastology_views)]
    print("✅ 有效视图:", view_df.predicted_view.unique())

    # ====================== 质量控制 ======================
    print(f"\n🔹 质量评估 ({len(view_df)} 个文件)")

    # 仅保留视频质量模型
    video_quality_model = model_utils.load_quality_classifier(
        input_type='video',
        weights_path=weights_dir / 'quality_control/weights/quality_control_model.pt'
    )

    # 所有文件统一使用视频质量评估逻辑
    if len(video_dataset) > 0:
        max_frames = max([v.shape[0] for v in video_dataset.values()])
        videos = model_utils.pad(list(video_dataset.values()), max_frames)
        videos = [x.permute(1, 0, 2, -1) for x in videos]
        video_quality_input = torch.stack(videos)
        pred_vid_qual = model_utils.quality_inference(video_quality_input, video_quality_model, list(video_dataset.keys()))
        quality_df = pd.DataFrame({
            'filename': list(pred_vid_qual.keys()),
            'pred_quality': list(pred_vid_qual.values())
        })
        video_quality_model.to("cpu")
    else:
        # 无视频文件时，质量评分默认设为满分通过
        quality_files = list(image_dataset.keys())
        quality_df = pd.DataFrame({
            'filename': quality_files,
            'pred_quality': [1.0 for _ in quality_files]
        })

    diastology = pd.merge(view_df, quality_df, on='filename')
    if to_save:
        diastology.to_csv(save_path / 'predicted_quality.csv', index=None)

    diastology = diastology[diastology.pred_quality >= quality_threshold]
    low_qual = quality_df[quality_df.pred_quality < quality_threshold].filename
    for f in low_qual:
        video_dataset.pop(f, None)
        image_dataset.pop(f, None)
    print(f"✅ 高质量文件: {len(diastology)} 个")

    # ====================== LVEF 计算 ======================
    print("\n🔹 计算 LVEF")
    a4c = diastology[diastology.predicted_view.isin(['BMode_A4C', 'BMode_A4C_LV'])]
    lvef_list = []
    if len(a4c) > 0:
        ef_model, ef_checkpoint = model_utils.ef_regressor()
        for f in a4c.filename:
            if f in video_dataset:
                a4c_tensor = video_dataset[f]
                ef = model_utils.predict_lvef(a4c_tensor, ef_model, ef_checkpoint)
                lvef_list.append((ef, f))
            else:
                print(f"⚠️ 非视频文件，无法计算 LVEF: {f}")
        ef_model.to("cpu")
    lvef = pd.DataFrame(lvef_list, columns=['LVEF', 'filename']) if lvef_list else pd.DataFrame({'LVEF': [0], 'filename': ['']})

    # ====================== 左心房容积 LAVi ======================
    print("\n🔹 计算 LAVi")
    la_model = model_utils.load_la_model()
    a2c = diastology[diastology.predicted_view == 'BMode_A2C']
    lav = 0.0
    lavi = 0.0

    if len(a2c) == 0 and len(a4c) == 0:
        df_lavi = pd.DataFrame({'filename': [''], 'LAVi': [0], 'LAV': [0], 'BSA': [bsa]})
    else:
        if len(a2c) == 0 and len(a4c) > 0:
            left_atrial_volume = {}
            for f in a4c.filename:
                try:
                    a4c_tensor = video_dataset[f]
                    mask, area = model_utils.la_seg_inf(la_model, a4c_tensor)
                    left_atrial_volume[f] = model_utils.calc_lav_from_a4c(mask, area)
                except:
                    left_atrial_volume[f] = 0.0
            if left_atrial_volume:
                a4c_key = max(left_atrial_volume, key=left_atrial_volume.get)
                lav = left_atrial_volume[a4c_key] * scale
                lavi = lav / bsa
                df_lavi = pd.DataFrame({'filename': [a4c_key], 'LAVi': [lavi], 'LAV': [lav], 'BSA': [bsa]})
        else:
            a4c_areas = {}
            a2c_areas = {}
            for f in a4c.filename:
                try:
                    t = video_dataset[f]
                    m, a = model_utils.la_seg_inf(la_model, t)
                    a4c_areas[np.max(a)] = (m, a, str(f))
                except:
                    continue
            for f in a2c.filename:
                try:
                    t = video_dataset[f]
                    m, a = model_utils.la_seg_inf(la_model, t)
                    a2c_areas[np.max(a)] = (m, a, str(f))
                except:
                    continue

            if a4c_areas and a2c_areas:
                a4c_best = a4c_areas[max(a4c_areas)]
                a2c_best = a2c_areas[max(a2c_areas)]
                lav = model_utils.calc_lav_biplane(a4c_best[0], a4c_best[1], a2c_best[0], a2c_best[1]) * scale
                lavi = lav / bsa
                df_lavi = pd.DataFrame({
                    'filename': [a4c_best[-1], a2c_best[-1]],
                    'LAVi': [lavi, lavi], 'LAV': [lav, lav], 'BSA': [bsa, bsa]
                })
            else:
                a4c_best = a4c_areas[max(a4c_areas)]
                lav = model_utils.calc_lav_from_a4c(a4c_best[0], a4c_best[1]) * scale
                lavi = lav / bsa
                df_lavi = pd.DataFrame({'filename': [a4c_best[-1]], 'LAVi': [lavi], 'LAV': [lav], 'BSA': [bsa]})
    la_model.to("cpu")

    # ====================== 多普勒测量 ======================
    print("\n🔹 测量多普勒参数")
    doppler_results = []
    for key, views in doppler_names.items():
        m_df = diastology[diastology.predicted_view.isin(views)]
        if m_df.empty:
            doppler_results.append(pd.DataFrame({key: [0]}))
            continue
        res = []
        for f in m_df.filename:
            dcm_path = path / f
            img, pv, x, y = model_utils.doppler_inference(dcm_path, key.lower())
            res.append((f, pv if pv != -1 else 0, x, y))
            if to_save and (pv, x, y) != (0, 0, 0):
                d = save_path / f'{key}_results'
                d.mkdir(exist_ok=True)
                dicom_utils.plot_results(key, dcm_path, pv, x, y, d)
        doppler_results.append(pd.DataFrame(res, columns=['filename', key, 'x', 'y'])[['filename', key]])

    doppler_measurements = pd.concat(doppler_results, ignore_index=True)

    eovera_df = diastology[diastology.predicted_view == 'DOPPLER_A4C_MV_PW']
    if eovera_df.empty:
        ea_vel = pd.DataFrame({'filename': [''], 'MV_E_over_A': [0], 'MV_E': [0], 'MV_A': [0]})
    else:
        ea_list = []
        for f in eovera_df.filename:
            dcm_path = path / f
            img, y0, x1, x2, y1, y2, a, e, ea = model_utils.eovera_inference(dcm_path)
            ea_list.append((f, ea, e, a, y0, x1, x2, y1, y2))
            if to_save and (x1, x2, y1, y2) != (0, 0, 0, 0):
                d = save_path / 'mvEoverA_results'
                d.mkdir(exist_ok=True)
                dicom_utils.plot_results('MV E/A', dcm_path, ea, x1, y1 + y0, d, x2, y2 + y0)
        ea_vel = pd.DataFrame(ea_list, columns=['filename', 'MV_E_over_A', 'MV_E', 'MV_A', 'y0', 'x1', 'x2', 'y1', 'y2'])

    # ====================== 舒张功能分级 ======================
    print(f"\n🔹 按照 {ase_year} ASE 指南分级")
    params = diastology.copy()
    params = pd.merge(params, lvef[['filename', 'LVEF']], on='filename', how='outer')
    params = pd.concat([params, doppler_measurements], ignore_index=True)
    params = pd.concat([params, ea_vel[['filename', 'MV_E_over_A', 'MV_E', 'MV_A']]], ignore_index=True)
    params = pd.concat([params, df_lavi], ignore_index=True)

    lvef_val = params['LVEF'].mean() if 'LVEF' in params else 0
    lavi_val = lavi
    medevel = params['MEDEVEL'].mean() if 'MEDEVEL' in params else 100
    latevel = params['LATEVEL'].mean() if 'LATEVEL' in params else 100
    trvmax = params['TRVMAX'].max() if 'TRVMAX' in params else 0
    mvA = params['MV_A'].mean() if 'MV_A' in params else 0
    mvE = params['MV_E'].mean() if 'MV_E' in params else 0
    mvEoverA = params['MV_E_over_A'].mean() if 'MV_E_over_A' in params else 0
    mvE_eprime = ase_guidelines.calc_eeprime(mvE, latevel, medevel) if mvE > 0 else 0

    print(f"💡 LVEF: {lvef_val:.2f}")
    print(f"💡 LAVi: {lavi_val:.2f}")
    print(f"💡 Medial e': {medevel:.2f}")
    print(f"💡 Lateral e': {latevel:.2f}")
    print(f"💡 TR Vmax: {trvmax:.2f}")
    print(f"💡 MV E: {mvE:.2f}")
    print(f"💡 MV E/e': {mvE_eprime:.2f}")
    print(f"💡 MV E/A: {mvEoverA:.2f}")

    if ase_year == 2016:
        if lvef_val >= 50:
            grade = ase_guidelines.preserved_ef_dd(medevel, latevel, trvmax, mvE_eprime, lavi_val)
            if grade == 1:
                grade = ase_guidelines.reduced_ef_dd(trvmax, mvE_eprime, mvEoverA, mvE, lavi_val)
        else:
            grade = ase_guidelines.reduced_ef_dd(trvmax, mvE_eprime, mvEoverA, mvE, lavi_val)
    else:
        grade = ase_guidelines.ase2025(medevel, latevel, trvmax, lavi_val, mvEoverA, mvE)

    diastolic_grade = ase_guidelines.map_grade_to_text[grade]
    print(f"\n✅ 分级结果: {diastolic_grade}")

    # ====================== 保存结果 ======================
    if to_save:
        params.to_csv(save_path / 'diastology.csv', index=False)
        files_str = ';'.join([str(f) for f in params.filename.dropna().unique()])
        views_str = ';'.join(params.predicted_view.dropna().unique())
        study_df = pd.DataFrame({
            'LVEF': lvef_val, 'LAV': lav, 'LAVi': lavi_val, 'TRVmax': trvmax,
            'MEDEVEL': medevel, 'LATEVEL': latevel, 'MV_E': mvE, 'MV_A': mvA,
            'MV_E_eprime': mvE_eprime, 'MV_E_over_A': mvEoverA,
            'diastology_grade': diastolic_grade, 'numeric_diastology_grade': grade,
            'views': views_str, 'files': files_str
        }, index=[0])
        study_df.to_csv(save_path / 'study_level_diastology.csv', index=False)

    print("\n🎉 舒张功能分析全部完成！")

# ====================== Click 命令行（框架标准） ======================
@click.command("automate_diastology")
@click.option("--path", required=True, type=str, help="DICOM 研究目录路径")
@click.option("--guideline_year", required=True, type=int, help="ASE 指南年份：2016 / 2025")
@click.option("--quality_threshold", default=0.0, type=float, help="最低质量阈值")
@click.option("--to_save", default=True, type=bool, help="是否保存结果")
@click.option("--save_path", default=None, type=str, help="结果保存路径")
def run(path, guideline_year, quality_threshold, to_save, save_path):
    run_diastology_pipeline(
        path=path,
        guideline_year=guideline_year,
        quality_threshold=quality_threshold,
        to_save=to_save,
        save_path=save_path
    )

# ====================== 引擎注册（EchoSky 必备） ======================
def register():
    return {
        "name": "automate_diastology",
        "entry": run,
        "description": "心脏舒张功能全自动评估（LVEF+LAVi+多普勒+ASE分级）"
    }

if __name__ == "__main__":
    run()