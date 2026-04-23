import subprocess
import sys

class CardiacEchoEngine:
    """
    心脏超声功能选择执行器
    不自动流水线！想跑什么功能就跑什么
    """

    def __init__(self):
        # 注册所有功能 → 对应执行命令
        self.commands = {
            "lv_segmentation_train": [
                sys.executable, "modules/segmentation/lv_segmentation.py",
                "--data_dir", "data",
                "--num_epochs", "50",
                "--batch_size", "8",
                "--device", "cuda"
            ],
            "lv_segmentation_test": [
                sys.executable, "modules/segmentation/lv_segmentation.py",
                "--data_dir", "data",
                "--weights", "output/segmentation/best.pt",
                "--run_test",
                "--device", "cuda"
            ],
            "lv_segmentation_video": [
                sys.executable, "modules/segmentation/lv_segmentation.py",
                "--data_dir", "data",
                "--weights", "output/segmentation/best.pt",
                "--save_video",
                "--device", "cuda"
            ]
        }

    def run(self, task_name):
        """
        🔥 选择功能直接运行！
        可选：
        - lv_segmentation_train
        - lv_segmentation_test
        - lv_segmentation_video
        """
        if task_name not in self.commands:
            raise ValueError(f"未知功能：{task_name}")
        
        print(f"🚀 正在运行：{task_name}")
        subprocess.run(self.commands[task_name])


# ======================
# 你只需要这样用！
# ======================
if __name__ == "__main__":
    engine = CardiacEchoEngine()

    # 选择你要执行的功能（三选一）
    # engine.run("lv_segmentation_train")    # 训练
    # engine.run("lv_segmentation_test")     # 测试
    engine.run("lv_segmentation_video")    # 推理+保存视频