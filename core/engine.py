import sys
import subprocess
from importlib import import_module

class CardiacEchoEngine:
    def __init__(self):
        self.modules = {}
        self._discover_modules()

    def _discover_modules(self):
        """自动发现并注册所有功能模块"""
        module_paths = [
            "modules.segmentation.lv_segmentation",
            "modules.functional_analysis.lv_ef_prediction",
            "modules.measurement.plax_hypertrophy_inference",
            "modules.disease_classification.a4c_classification_inference",
        ]

        for path in module_paths:
            mod = import_module(path)
            info = mod.register()
            self.modules[info["name"]] = {
                "path": path.replace(".", "/") + ".py",  # 自动得到文件路径
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

        # 自动把关键字参数 → click 命令行参数
        for key, value in kwargs.items():
            if isinstance(value, bool):
                # 处理 True/False
                if value:
                    cmd.append(f"--{key}")
                else:
                    cmd.append(f"--skip_{key}")
            else:
                cmd.append(f"--{key}")
                cmd.append(str(value))

        print(f"🚀 正在执行：{' '.join(cmd)}")
        subprocess.run(cmd)