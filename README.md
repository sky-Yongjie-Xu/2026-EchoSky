# EchoSky

**EchoSky** 是一个基于深度学习的超声心动图（心脏超声）智能分析系统。该项目集成了多种先进的计算机视觉模型，用于实现心脏超声视频的自动化分析，包括视角分类、左心室分割、射血分数预测、自动测量、疾病分类以及结构化报告生成等功能。

## 📋 功能特性

### ✅ 已实现功能

| 模块 | 功能描述 | 模型架构 |
|------|----------|----------|
| **视角分类** | 自动识别超声切面类型（A2C, A3C, A4C, A5C, PLAX, PSAX等11种） | ConvNeXt-Base |
| **Subcostal视图分类** | 肋下视图分类（高质量筛选 Step1） | R(2+1)D-18 |
| **质量控制** | Subcostal图像质量控制（高质量筛选 Step2） | R(2+1)D-18 |
| **左心室分割** | 对左心室进行像素级分割，支持训练、测试和视频生成 | DeepLabV3+/FCN |
| **射血分数预测** | 预测左心室射血分数（LVEF），支持多clip推理 | R(2+1)D-18 |
| **报告生成（EchoPrime）** | 基于EchoPrime架构，自动生成结构化超声报告（支持中英文） | MViT-V2 + ConvNeXt |
| **报告生成（EchoGemma）** | 基于Gemma的超声智能报告生成 | Gemma |
| **B模式线性测量** | 2D结构分割测量（IVS, LVID, LVPW, Aorta, LA, RV, PA, IVC） | DeepLabV3-ResNet50 |
| **多普勒峰值速度测量** | 多普勒超声峰值速度测量（AVVmax, TRVmax, MRVmax, LVOTVmax等） | DeepLabV3-ResNet50 |
| **二尖瓣E/A测量** | 二尖瓣血流多普勒 E峰/A峰 速度测量及E/A比值计算 | DeepLabV3-ResNet50 |
| **TAPSE测量** | 三尖瓣环收缩期位移（TAPSE）测量，评估右心室功能 | DeepLabV3-ResNet50 |
| **肝脏疾病预测** | 基于超声图像预测肝脏疾病（肝硬化/脂肪肝） | DenseNet-121 |
| **年龄预测** | 基于超声视频预测年龄 | R(2+1)D-18 |
| **视觉问答** | 超声视频多选题评估（MedGemma 1.5） | MedGemma-1.5-4B |
| **PLAX自动测量** | 在PLAX视角下自动测量LVPW、LVID、IVS等指标 | DeepLabV3-ResNet50 |
| **疾病分类** | A4C视角下的淀粉样变性二分类 | R3D-18 |

### 🚧 计划开发功能

-  landmarks检测
-  更多疾病分类模型

## 🏗️ 项目结构

```
EchoSky/
├── main.py                          # 主入口文件
├── core/
│   └── engine.py                    # 核心引擎，负责模块加载和调度
├── data/
│   └── echo.py                      # EchoNet-Dynamic 数据集加载器
├── modules/
│   ├── view_classification/         # 视角分类模块
│   │   ├── view_classification_echoprime.py    # EchoPrime视角分类
│   │   ├── subcostal_view_classification.py    # Subcostal视图分类
│   │   └── utils.py
│   ├── segmentation/                # 左心室分割模块
│   │   └── lv_segmentation_dynamic.py
│   ├── functional_analysis/         # 射血分数预测模块
│   │   └── lv_ef_prediction_dynamic.py
│   ├── report_generation/           # 报告生成模块
│   │   ├── report_generation_echoprime.py    # EchoPrime报告生成
│   │   ├── report_generation_gemma.py        # EchoGemma报告生成
│   │   └── utils.py
│   ├── measurement/                 # 自动测量模块
│   │   ├── b_mode_linear_measurement.py      # B模式2D结构测量
│   │   ├── doppler_measurement.py            # 多普勒峰值速度测量
│   │   ├── doppler_mv_ea_measurement.py      # 二尖瓣E/A测量
│   │   ├── doppler_tapse_measurement.py      # TAPSE测量
│   │   ├── plax_hypertrophy_inference.py     # PLAX测量（待启用）
│   │   └── utils.py
│   ├── disease_classification/      # 疾病分类模块
│   │   ├── liver_disease_prediction.py       # 肝脏疾病预测
│   │   ├── a4c_classification_inference.py   # 淀粉样变分类（待启用）
│   │   └── utils.py
│   ├── quality_control/             # 质量控制模块
│   │   ├── subcostal_quality_control.py      # Subcostal质量控制
│   │   └── utils.py
│   ├── age_prediction/              # 年龄预测模块
│   │   ├── age_prediction.py                 # 超声年龄预测
│   │   └── utils.py
│   ├── visual_question_answering/   # 视觉问答模块
│   │   └── visual_question_answering.py      # MedGemma VQA评估
│   └── landmark_detection/          # 地标检测模块（待开发）
├── configs/
│   └── train_config.yaml            # 训练配置文件
├── weights/                         # 模型权重（需单独下载）
│   ├── 2D_models/                   # B模式测量模型
│   └── Doppler_models/              # 多普勒测量模型
└── README.md
```

## 🚀 快速开始

### 1. 环境要求

- Python >= 3.8
- PyTorch >= 1.9
- torchvision
- echonet
- OpenCV
- matplotlib
- pandas
- numpy
- scikit-learn
- transformers (用于报告生成)
- pydicom

### 2. 安装依赖

```bash
pip install torch torchvision torchaudio
pip install echonet opencv-python matplotlib pandas numpy scikit-learn
pip install transformers pydicom tqdm click
```

### 3. 基本使用

#### 查看所有可用功能

```python
python main.py
```

#### 运行特定模块

```python
from core.engine import CardiacEchoEngine

engine = CardiacEchoEngine()

# 左心室分割（带视频可视化）
engine.run("lv_segmentation_dynamic", save_video=True)

# 视角分类
engine.run("view_classification_echoprime", dataset_dir="path/to/dicom/folder", visualize=True)

# Subcostal视图分类（高质量筛选 Step1）
engine.run("subcostal_view_classification", dataset="path/to/dataset", manifest_path="path/to/manifest.csv")

# 质量控制（高质量筛选 Step2）
engine.run("subcostal_quality_control", dataset="path/to/dataset", manifest_path="path/to/manifest.csv")

# 肝脏疾病预测（支持肝硬化/脂肪肝）
engine.run("liver_disease_prediction", dataset="path/to/dataset", manifest_path="path/to/manifest.csv", label="cirrhosis")

# 射血分数预测
engine.run("lv_ef_prediction_dynamic")

# B模式线性测量（支持IVS, LVID, LVPW, Aorta, LA, RV, PA, IVC）
engine.run("b_mode_linear_measurement", model_weights="aorta", folders="path/to/videos", output_path_folders="output/measurement")

# 多普勒峰值速度测量（支持AVVmax, TRVmax, MRVmax, LVOTVmax等）
engine.run("doppler_measurement", model_weights="avvmax", folders="path/to/videos", output_path_folders="output/doppler")

# 二尖瓣E/A测量（计算E峰、A峰速度及E/A比值）
engine.run("doppler_mv_ea_measurement", folders="path/to/videos", output_path_folders="output/mv_ea")

# TAPSE测量（三尖瓣环收缩期位移，评估右心室功能）
engine.run("doppler_tapse_measurement", folders="path/to/videos", output_path_folders="output/tapse")

# 报告生成（EchoPrime，支持中英文）
engine.run("report_generation_echoprime", dataset_dir="path/to/dicom/folder")

# 报告生成（EchoGemma，基于Gemma的智能报告）
engine.run("report_generation_gemma", dicom_dir="path/to/dicom/folder", save_path="output/report_gemma.txt")

# 年龄预测（基于超声视频）
engine.run("age_prediction", target="Age", manifest_path="path/to/manifest.csv", path_column="video_path", weights_path="path/to/weights.pt", save_path="output/predictions.csv")

# 视觉问答（MedGemma 1.5 多选题评估）
engine.run("visual_question_answering", dataset_dir="path/to/dataset", manifest_path="path/to/manifest.csv", output_path="output/vqa_results.json")

# PLAX自动测量（待启用）
# engine.run("plax_inference", in_dir="path/to/videos", out_dir="output/plax")

# 淀粉样变分类（待启用）
# engine.run("a4c_classification", in_dir="path/to/videos", out_dir="output/a4c")
```

## 📊 数据准备

### 数据集格式

本项目主要基于 **EchoNet-Dynamic** 数据集格式：

```
dataset_root/
├── Videos/              # 超声视频文件（.avi格式）
├── FileList.csv         # 文件列表和标签
├── VolumeTracings.csv   # 心室容积描记数据
└── ...
```

### DICOM数据

对于视角分类和报告生成模块，支持直接读取DICOM格式数据：

```
dicom_folder/
├── study1/
│   ├── *.dcm
│   └── ...
└── study2/
    ├── *.dcm
    └── ...
```

## ⚙️ 配置说明

训练配置文件位于 `configs/train_config.yaml`，主要参数包括：

```yaml
training:
  modules:
    - name: "segmentation"
      enabled: true
      model: "unet_plusplus"
      loss: "dice_focal"
    
    - name: "landmark_detection"
      enabled: false
      model: "hrnet"
      loss: "mse"

  data:
    augmentation: "echo_specific_aug"
    batch_size: 8

  optimizer:
    type: "adamw"
    lr: 1e-4
```

## 🔧 模块详细说明

### 1. 视角分类 (View Classification)

- **输入**: DICOM视频文件夹
- **输出**: 每个视频对应的视角类别
- **支持视角**: A2C, A3C, A4C, A5C, Apical_Doppler, Doppler_Parasternal_Long, Doppler_Parasternal_Short, Parasternal_Long, Parasternal_Short, SSN, Subcostal

### 2. 左心室分割 (LV Segmentation)

- **输入**: EchoNet-Dynamic格式数据集
- **输出**: 
  - 分割模型权重
  - 带有分割结果的视频（可选）
  - 每个帧的左心室面积曲线
- **评估指标**: Dice系数（整体、舒张期、收缩期）

### 3. Subcostal视图分类 (Subcostal View Classification)

- **输入**: 数据集路径和manifest文件
- **输出**: CSV文件（包含预测结果，筛选出高质量Subcostal视图）
- **临床意义**: 高质量筛选流程的第一步（Step1）
- **使用示例**:
  ```python
  engine.run("subcostal_view_classification", dataset="path/to/dataset", manifest_path="path/to/manifest.csv")
  ```

### 4. 质量控制 (Quality Control)

- **输入**: 数据集路径和manifest文件
- **输出**: CSV文件（包含质量控制评分，筛选出高质量图像）
- **临床意义**: 高质量筛选流程的第二步（Step2）
- **使用示例**:
  ```python
  engine.run("subcostal_quality_control", dataset="path/to/dataset", manifest_path="path/to/manifest.csv")
  ```

### 5. 射血分数预测 (EF Prediction)

- **输入**: EchoNet-Dynamic格式数据集
- **输出**:
  - 预测的LVEF值
  - 散点图、ROC曲线
  - R²、MAE、RMSE等评估指标
- **支持多clip推理**，自动聚合预测结果

### 6. 报告生成 (Report Generation)

- **输入**: DICOM视频文件夹
- **输出**: 结构化文本报告
- **支持语言**: 中文(zh) / 英文(en)
- **报告章节**: 左心室、右心室、左心房、右心房、瓣膜等

### 7. B模式线性测量 (B-Mode Linear Measurement)

- **输入**: 视频文件夹（.avi 或 .dcm 格式）
- **输出**:
  - 带标注的视频文件
  - CSV文件（包含坐标和物理测量值）
- **支持测量**:
  - **IVS**: 室间隔厚度
  - **LVID**: 左心室内径
  - **LVPW**: 左室后壁厚度
  - **Aorta**: 主动脉内径
  - **Aortic Root**: 主动脉根部
  - **LA**: 左心房内径
  - **RV Base**: 右心室基底段
  - **PA**: 肺动脉内径
  - **IVC**: 下腔静脉内径
- **使用示例**:
  ```python
  engine.run("b_mode_linear_measurement", model_weights="aorta", folders="path/to/videos", output_path_folders="output/measurement")
  ```

### 8. 多普勒峰值速度测量 (Doppler Peak Velocity Measurement)

- **输入**: DICOM多普勒图像文件夹
- **输出**:
  - 标注峰值位置的图像
  - CSV文件（包含峰值速度值）
- **支持测量**:
  - **AVVmax**: 主动脉瓣最大速度
  - **TRVmax**: 三尖瓣反流最大速度
  - **MRVmax**: 二尖瓣反流最大速度
  - **LVOTVmax**: 左室流出道最大速度
  - **LateVel**: 晚期充盈速度
  - **MeDvel**: 平均舒张期速度
- **使用示例**:
  ```python
  engine.run("doppler_measurement", model_weights="avvmax", folders="path/to/videos", output_path_folders="output/doppler")
  ```

### 9. 二尖瓣E/A测量 (Mitral Valve E/A Measurement)

- **输入**: DICOM二尖瓣血流多普勒图像文件夹
- **输出**:
  - 标注E峰和A峰位置的图像
  - CSV文件（包含E峰速度、A峰速度、E/A比值）
- **临床意义**: 评估左心室舒张功能
- **测量参数**:
  - **E峰速度**: 早期充盈速度
  - **A峰速度**: 晚期充盈速度（心房收缩）
  - **E/A比值**: 舒张功能评估指标
- **使用示例**:
  ```python
  engine.run("doppler_mv_ea_measurement", folders="path/to/videos", output_path_folders="output/mv_ea")
  ```

### 10. TAPSE测量 (Tricuspid Annular Plane Systolic Excursion)

- **输入**: DICOM三尖瓣多普勒图像文件夹
- **输出**:
  - 标注测量点的图像
  - CSV文件（包含TAPSE值）
- **临床意义**: 评估右心室收缩功能的重要指标
- **测量参数**:
  - **TAPSE**: 三尖瓣环收缩期位移（单位：cm）
- **使用示例**:
  ```python
  engine.run("doppler_tapse_measurement", folders="path/to/videos", output_path_folders="output/tapse")
  ```

### 11. PLAX自动测量 (PLAX Measurement) ⚠️ 待启用

- **输入**: PLAX视角视频文件夹
- **输出**:
  - CSV文件（包含每帧的测量值）
  - 带标注的视频
  - 测量曲线图
- **测量指标**: LVPW（左室后壁厚度）、LVID（左室内径）、IVS（室间隔厚度）

### 12. 肝脏疾病预测 (Liver Disease Prediction)

- **输入**: 数据集路径和manifest文件
- **输出**: CSV文件（包含预测结果）
- **支持疾病**:
  - **Cirrhosis**: 肝硬化预测
  - **SLD**: 脂肪肝（Steatotic Liver Disease）预测
- **模型架构**: DenseNet-121
- **使用示例**:
  ```python
  engine.run("liver_disease_prediction", dataset="path/to/dataset", manifest_path="path/to/manifest.csv", label="cirrhosis")
  ```

### 13. 疾病分类 (Disease Classification) ⚠️ 待启用

- **输入**: A4C视角视频文件夹
- **输出**: CSV文件（包含每个视频的阳性置信度）
- **当前支持**: 淀粉样变性分类

## 📝 许可证

本项目仅供学术研究使用。

## 🙏 致谢

- **EchoNet-Dynamic**: 提供大规模超声心动图数据集
- **EchoPrime**: 提供报告生成模型架构
- **PyTorch**: 深度学习框架

## 📧 联系方式

如有问题或建议，请通过 GitHub Issues 联系。

---

**注意**: 本系统为研究工具，不适用于临床诊断。所有结果应由专业医师审核。