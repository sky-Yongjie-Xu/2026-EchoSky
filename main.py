from core.engine import CardiacEchoEngine

if __name__ == "__main__":
    engine = CardiacEchoEngine()
    engine.list_modules()  # 自动打印所有可用功能

    #### 左心室 分割
    # engine.run("lv_segmentation", save_video=True)

    #### 左心室 射血分数预测
    # engine.run("lv_ef_prediction")

    #### PLAX 心脏超声自动测量（LVPW、LVID、IVS）
    # engine.run("plax_inference", in_dir="a4c-video-dir/Videos", out_dir="output/plax_inference")

    #### A4C 疾病分类
    engine.run("a4c_classification", in_dir="a4c-video-dir/Videos", out_dir="output/a4c_classification")