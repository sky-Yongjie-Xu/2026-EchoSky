import sys
import os
import subprocess
from importlib import import_module

class CardiacEchoEngine:
    def __init__(self):
        self.modules = {}
        self._fix_module_imports()  # ✅ 先修复导入
        self._discover_modules()

    def _fix_module_imports(self):
        """
        🔥 核心修复：自动把所有模块文件夹加入 sys.path
        所有子模块都可以直接 import 同级文件
        """
        base = os.path.abspath(".")
        module_dirs = [
            "modules/segmentation",
            "modules/functional_analysis",
            "modules/classification",
            "modules/measurement",
            "modules/disease_classification",
        ]
        for d in module_dirs:
            p = os.path.join(base, d)
            if p not in sys.path:
                sys.path.insert(0, p)

    def _discover_modules(self):
        """自动发现并注册所有功能模块"""
        module_paths = [
            "modules.classification.view_classification_echoprime",
            "modules.segmentation.lv_segmentation_dynamic",
            "modules.functional_analysis.lv_ef_prediction_dynamic",
            "modules.measurement.b_mode_linear_measurement",
            "modules.measurement.doppler_measurement",
            "modules.report_generation.report_generation_echoprime",


            # "modules.measurement.plax_hypertrophy_inference",
            # "modules.disease_classification.a4c_classification_inference",
        ]

        for path in module_paths:
            mod = import_module(path)
            info = mod.register()
            self.modules[info["name"]] = {
                "path": path.replace(".", "/") + ".py",
                "desc": info["description"]
            }

    def list_modules(self):
        print("📋 可用功能列表：")
        for i, (name, info) in enumerate(self.modules.items(), 1):
            print(f"{i}. {name}: {info['desc']}")

    def run(self, task_name, **kwargs):
        """执行指定功能，自动转成 click 命令行运行"""
        if task_name not in self.modules:
            raise ValueError(f"未知功能：{task_name}，可用：{list(self.modules.keys())}")

        script_path = self.modules[task_name]["path"]
        cmd = [sys.executable, script_path]

        for key, value in kwargs.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.append(f"--{key}")
                cmd.append(str(value))

        print(f"🚀 正在执行：{' '.join(cmd)}")
        subprocess.run(cmd)