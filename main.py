from core.engine import CardiacEchoEngine

if __name__ == "__main__":
    engine = CardiacEchoEngine()
    engine.list_modules()  # 自动打印所有可用功能

    ###################### 已有功能 ######################

    #### 心脏视角分类
    # engine.run("view_classification_echoprime", dataset_dir="datasets/example_study_echoprime", visualize=True)

    #### 胸骨旁视图分类
    # engine.run("subcostal_view_classification", dataset="datasets/example_study_dynamic", manifest_path="datasets/manifest_step_1_and_2.csv")

    #### 质量控制
    # engine.run("subcostal_quality_control", dataset="datasets/example_study_dynamic", manifest_path="datasets/manifest_step_1_and_2.csv")

    #### liver_disease_prediction
    # engine.run("liver_disease_prediction", dataset="datasets/example_study_dynamic", manifest_path="datasets/manifest.csv", label="cirrhosis")

    #### 左心室 分割
    # engine.run("lv_segmentation_dynamic", save_video=True)

    #### 左心室 射血分数预测
    # engine.run("lv_ef_prediction_dynamic")

    #### 心脏B模式线性测量
    # engine.run("b_mode_linear_measurement", model_weights="aorta", folders="datasets/example_study_dynamic", output_path_folders="modules/measurement/output")

    ### 心脏多普勒测量
    # engine.run("doppler_measurement", model_weights="avvmax", folders="datasets/example_study_dynamic", output_path_folders="modules/measurement/output")
    # engine.run("doppler_mv_ea_measurement", folders="datasets/example_study_dynamic", output_path_folders="modules/measurement/output")
    # engine.run("doppler_tapse_measurement", folders="datasets/example_study_dynamic", output_path_folders="modules/measurement/output")

    #### 心脏结构化报告生成
    # engine.run("report_generation_echoprime", dataset_dir="datasets/example_study_echoprime")

    #### 心脏结构化报告生成
    # engine.run("report_generation_gemma", dicom_dir="datasets/example_study_echoprime", save_path="modules/report_generation/outputs/report_gemma.txt")
    
    #### 超声年龄预测（视频）
    # engine.run("age_prediction", target="Age", manifest_path="datasets/manifest_age.csv", path_column="video_path", weights_path="modules/age_prediction/weights/a2c_model_best_epoch_val_mae.pt", save_path="modules/age_prediction/outputs/predictions.csv")

    #### 视觉问答
    engine.run("visual_question_answering", media="datasets/example_study_dynamic/0X1A0A263B22CCD966.avi")
    # engine.run("visual_question_answering_echo", media="datasets/example_study_dynamic/0X1A0A263B22CCD966.avi")




    ###################### 计划开发功能（敬请期待） ######################

    #### PLAX 心脏超声自动测量（LVPW、LVID、IVS）
    # engine.run("plax_inference", in_dir="a4c-video-dir/Videos", out_dir="output/plax_inference")

    #### A4C 疾病分类
    # engine.run("a4c_classification", in_dir="a4c-video-dir/Videos", out_dir="output/a4c_classification")
