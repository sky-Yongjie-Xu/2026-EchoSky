# EchoSky

**EchoSky** 是一个基于深度学习的超声心动图（心脏超声）智能分析系统。该项目提供了从数据质控、图像分割、功能分析到疾病预测和报告生成的完整工作流程。

## 📋 功能特性

### ✅ 已实现功能（按推荐工作流程排序）

#### Step 1: 数据质控
| 模块 | 功能描述 | 模型架构 |
|------|----------|----------|
| **Subcostal视图分类** | 肋下视图分类（高质量筛选 Step1） | R(2+1)D-18 |
| **质量控制** | Subcostal图像质量控制（高质量筛选 Step2） | R(2+1)D-18 |

#### Step 2: 视角分类
| 模块 | 功能描述 | 模型架构 |
|------|----------|----------|
| **视角分类** | 自动识别超声切面类型（A2C, A3C, A4C, A5C, PLAX, PSAX等11种） | ConvNeXt-Base |

#### Step 3: 分割与测量
| 模块 | 功能描述 | 模型架构 |
|------|----------|----------|
| **左心室分割** | 对左心室进行像素级分割，支持训练、测试和视频生成 | DeepLabV3+/FCN |
| **B模式线性测量** | 2D结构分割测量（IVS, LVID, LVPW, Aorta, LA, RV, PA, IVC） | DeepLabV3-ResNet50 |
| **多普勒峰值速度测量** | 多普勒超声峰值速度测量（AVVmax, TRVmax, MRVmax, LVOTVmax等） | DeepLabV3-ResNet50 |
| **二尖瓣E/A测量** | 二尖瓣血流多普勒 E峰/A峰 速度测量及E/A比值计算 | DeepLabV3-ResNet50 |
| **TAPSE测量** | 三尖瓣环收缩期位移（TAPSE）测量，评估右心室功能 | DeepLabV3-ResNet50 |

#### Step 4: 功能分析
| 模块 | 功能描述 | 模型架构 |
|------|----------|----------|
| **射血分数预测** | 预测左心室射血分数（LVEF），支持多clip推理 | R(2+1)D-18 |
| **年龄预测** | 基于超声视频预测年龄 | R(2+1)D-18 |
| **全自动舒张功能评估** | 端到端舒张功能分析，包含视图分类、质量控制、LVEF计算、左心房容积(LAVi)测量、多普勒参数提取，自动按照 ASE 2016 / 2025 指南分级 | 多模型集成流水线 |

#### Step 5: 疾病预测（可选）
| 模块 | 功能描述 | 模型架构 |
|------|----------|----------|
| **肝脏疾病预测** | 基于超声图像预测肝脏疾病（肝硬化/脂肪肝） | DenseNet-121 |
| **二尖瓣疾病预测** | 全自动二尖瓣病变分级（无/轻度/中度/重度）+ 风湿性心脏病二分类 + 平均压差预测，多视图集成推理 | R(2+1)D-18 + 梯度提升集成模型 |

#### Step 6: 报告生成
| 模块 | 功能描述 | 模型架构 |
|------|----------|----------|
| **报告生成（EchoPrime）** | 基于EchoPrime架构，自动生成结构化超声报告（支持中英文） | MViT-V2 + ConvNeXt |
| **报告生成（EchoGemma）** | 基于Gemma的超声智能报告生成 | Gemma 2B/7B |

#### Step 7: 智能问答（基于报告）
| 模块 | 功能描述 | 模型架构 |
|------|----------|----------|
| **视觉问答（Echo专用）** | 超声领域专用视觉问答系统 | 多模态融合模型 |
| **视觉问答（MedGemma）** | 基于生成的报告和图像进行多选题评估 | MedGemma-1.5-4B |

#### 待启用功能
| 模块 | 功能描述 | 模型架构 |
|------|----------|----------|
| **PLAX自动测量** | 在PLAX视角下自动测量LVPW、LVID、IVS等指标 | DeepLabV3-ResNet50 |
| **疾病分类** | A4C视角下的淀粉样变性二分类 | R3D-18 |

### 🚧 计划开发功能
- landmarks检测
- 更多疾病分类模型
- 多模态融合诊断
- 实时视频流分析

## 🔄 推荐工作流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                        超声数据分析工作流程                           │
└─────────────────────────────────────────────────────────────────────┘

Step 1: 数据质控
┌──────────────────────┐    ┌──────────────────────┐
│  Subcostal视图分类    │ -> │    质量控制          │
│  (筛选正确视图)       │    │  (筛选高质量图像)     │
└──────────────────────┘    └──────────────────────┘
                ↓
Step 2: 视角分类
┌──────────────────────┐
│    视角分类           │
│  (识别切面类型)       │
└──────────────────────┘
                ↓
Step 3: 分割与测量
┌──────────────────────┐    ┌──────────────────────┐
│   左心室分割          │    │  B模式线性测量        │
│  (提取LV轮廓)        │    │  (测量IVS/LVID/LVPW) │
└──────────────────────┘    └──────────────────────┘
                ↓                         ↓
┌──────────────────────┐    ┌──────────────────────┐
│  多普勒峰值速度测量   │    │   二尖瓣E/A测量       │
│  (AVVmax/TRVmax等)   │    │  (评估舒张功能)       │
└──────────────────────┘    └──────────────────────┘
                ↓
┌──────────────────────┐
│    TAPSE测量          │
│  (评估右心室功能)     │
└──────────────────────┘
                ↓
Step 4: 功能分析
┌──────────────────────┐    ┌──────────────────────┐
│  射血分数预测         │    │    年龄预测          │
│  (LVEF计算)          │    │  (基于超声视频)       │
└──────────────────────┘    └──────────────────────┘
                ↓
┌──────────────────────────────────────────────────┐
│          全自动舒张功能评估                      │
│  LVEF + LAVi + 多普勒测量 + ASE指南分级(2016/2025)│
└──────────────────────────────────────────────────┘
                ↓
Step 5: 疾病预测（可选）
┌──────────────────────┐    ┌──────────────────────┐
│  肝脏疾病预测         │    │  二尖瓣疾病预测       │
│  (肝硬化/脂肪肝)     │    │  (病变分级+风湿性)    │
└──────────────────────┘    └──────────────────────┘
                ↓
Step 6: 报告生成
┌──────────────────────┐    ┌──────────────────────┐
│  EchoPrime报告生成    │    │  EchoGemma报告生成   │
│  (结构化报告)        │    │  (智能报告)          │
└──────────────────────┘    └──────────────────────┘
                ↓
Step 7: 智能问答（基于报告）
┌──────────────────────┐
│    视觉问答           │
│  (MedGemma评估)      │
└──────────────────────┘
```

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
│   ├── quality_control/             # 质量控制模块
│   │   ├── subcostal_quality_control.py      # Subcostal质量控制
│   │   └── utils.py
│   ├── segmentation/                # 左心室分割模块
│   │   ├── lv_segmentation_dynamic.py
│   │   └── echonet/
│   │       ├── __init__.py
│   │       ├── __main__.py
│   │       ├── __version__.py
│   │       ├── config.py
│   │       └── utils/
│   │           ├── __init__.py
│   │           ├── segmentation.py
│   │           ├── video.py
│   │           └── video_original.py
│   ├── measurement/                 # 自动测量模块
│   │   ├── b_mode_linear_measurement.py      # B模式2D结构测量
│   │   ├── doppler_measurement.py            # 多普勒峰值速度测量
│   │   ├── doppler_mv_ea_measurement.py      # 二尖瓣E/A测量
│   │   ├── doppler_tapse_measurement.py      # TAPSE测量
│   │   ├── plax_hypertrophy_inference.py     # PLAX测量（待启用）
│   │   └── utils.py
│   ├── functional_analysis/         # 射血分数预测模块
│   │   └── lv_ef_prediction_dynamic.py
│   ├── disease_classification/      # 疾病分类模块
│   │   ├── liver_disease_prediction.py       # 肝脏疾病预测
│   │   ├── ms_disease_prediction.py          # 二尖瓣疾病+风湿性心脏病预测
│   │   ├── a4c_classification_inference.py   # 淀粉样变分类（待启用）
│   │   └── utils.py
│   ├── age_prediction/              # 年龄预测模块
│   │   ├── age_prediction.py                 # 超声年龄预测
│   │   └── utils.py
│   ├── visual_question_answering/   # 视觉问答模块
│   │   ├── visual_question_answering_echo.py      # Echo专用VQA实现
│   │   └── visual_question_answering_medgemma.py  # MedGemma VQA评估
│   ├── report_generation/           # 报告生成模块
│   │   ├── report_generation_echoprime.py    # EchoPrime报告生成
│   │   ├── report_generation_gemma.py        # EchoGemma报告生成
│   │   ├── echogemma/
│   │   │   ├── __init__.py
│   │   │   └── echogemma.py                 # EchoGemma核心实现
│   │   └── utils.py
│   ├── automate_diastology/         # 全自动舒张功能评估
│   │   ├── automate_diastology.py            # 主模块 端到端流水线
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── ase_guidelines.py             # ASE 2016 / 2025 指南实现
│   │       ├── dicom_utils.py                # DICOM 处理工具
│   │       ├── model_utils.py                # 模型加载与推理
│   │       └── lav_mask.py                   # 左心房分割
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

#### 完整工作流程示例

```python
from core.engine import CardiacEchoEngine

engine = CardiacEchoEngine()

# ========== 第一阶段：数据质控 ==========
# Step 1: Subcostal视图分类（高质量筛选 Step1）
engine.run("subcostal_view_classification", dataset="path/to/dataset", manifest_path="path/to/manifest.csv")

# Step 2: 质量控制（高质量筛选 Step2）
engine.run("subcostal_quality_control", dataset="path/to/dataset", manifest_path="path/to/manifest.csv")

# ========== 第二阶段：视角分类 ==========
# Step 3: 视角分类
engine.run("view_classification_echoprime", dataset_dir="path/to/dicom/folder", visualize=True)

# ========== 第三阶段：分割与测量 ==========
# Step 4: 左心室分割（带视频可视化）
engine.run("lv_segmentation_dynamic", save_video=True)

# Step 5: B模式线性测量
engine.run("b_mode_linear_measurement", model_weights="aorta", folders="path/to/videos", output_path_folders="output/measurement")

# Step 6: 多普勒峰值速度测量
engine.run("doppler_measurement", model_weights="avvmax", folders="path/to/videos", output_path_folders="output/doppler")

# Step 7: 二尖瓣E/A测量
engine.run("doppler_mv_ea_measurement", folders="path/to/videos", output_path_folders="output/mv_ea")

# Step 8: TAPSE测量
engine.run("doppler_tapse_measurement", folders="path/to/videos", output_path_folders="output/tapse")

# ========== 第四阶段：功能分析 ==========
# Step 9: 射血分数预测
engine.run("lv_ef_prediction_dynamic")

# Step 10: 年龄预测
engine.run("age_prediction", target="Age", manifest_path="path/to/manifest.csv", path_column="video_path", weights_path="path/to/weights.pt", save_path="output/predictions.csv")

# Step 11: 全自动舒张功能评估 (端到端流水线)
engine.run("automate_diastology", path="path/to/dicom/study", guideline_year=2025, save_path="output/diastology")

# ========== 第五阶段：疾病预测（可选） ==========
# Step 11: 肝脏疾病预测
engine.run("liver_disease_prediction", dataset="path/to/dataset", manifest_path="path/to/manifest.csv", label="cirrhosis")

# Step 12: 二尖瓣疾病预测（全自动DICOM推理）
engine.run("ms_disease_prediction", data_dir="path/to/dicom/studies", weights_dir="modules/disease_classification/weights", batch_size=4)

# ========== 第六阶段：报告生成 ==========
# Step 12: 报告生成（EchoPrime，支持中英文）
engine.run("report_generation_echoprime", dataset_dir="path/to/dicom/folder")

# Step 13: 报告生成（EchoGemma，基于Gemma的智能报告）
engine.run("report_generation_gemma", dicom_dir="path/to/dicom/folder", save_path="output/report_gemma.txt")

# ========== 第七阶段：智能问答（基于生成的报告） ==========
# Step 14: 视觉问答（基于报告的多选题评估）
engine.run("visual_question_answering", dataset_dir="path/to/dataset", manifest_path="path/to/manifest.csv", output_path="output/vqa_results.json")
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