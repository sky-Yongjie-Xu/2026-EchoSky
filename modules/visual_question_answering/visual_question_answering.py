# -*- coding: utf-8 -*-
import json
import os
import torch
import cv2
import re
import math
import click
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import Optional
from transformers import AutoProcessor, AutoModelForImageTextToText

# 自动加入项目根目录，确保导入正常
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ==============================================
# 模型加载
# ==============================================
def load_model(model_id: str):
    print(f"Loading model: {model_id}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    print("Model loaded successfully!")
    return model, processor


# ==============================================
# 视频帧提取
# ==============================================
def extract_frames_from_video(video_path, num_frames=1, strategy="middle"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise ValueError(f"Video has no frames: {video_path}")

    if strategy == "middle":
        indices = [total_frames // 2]
    elif strategy == "first":
        indices = [0]
    elif strategy == "uniform":
        if num_frames >= total_frames:
            indices = list(range(total_frames))
        else:
            step = total_frames / num_frames
            indices = [int(i * step) for i in range(num_frames)]
    else:
        indices = [total_frames // 2]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
    cap.release()
    return frames


def extract_and_concat_frames(video_paths, strategy="first"):
    frames = []
    for path in video_paths:
        frame_list = extract_frames_from_video(path, 1, strategy)
        if frame_list:
            frames.append(frame_list[0])
    if not frames:
        raise ValueError("No frames extracted")

    n = len(frames)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    max_w = max(f.width for f in frames)
    max_h = max(f.height for f in frames)
    grid = Image.new("RGB", (max_w * cols, max_h * rows))
    for i, img in enumerate(frames):
        r, c = divmod(i, cols)
        grid.paste(img, (c * max_w, r * max_h))
    return [grid]


# ==============================================
# Prompt 构建
# ==============================================
def format_multiple_choice_prompt(sample, include_report=False, multi_image=False):
    prompt_parts = []
    if include_report and sample.get("generated_report"):
        prompt_parts.append(f"Clinical Report:\n{sample['generated_report']}\n")
    prompt_parts.append(f"Question: {sample['question']}")
    prompt_parts.append("\nOptions:")
    for opt in ["A", "B", "C", "D"]:
        key = f"option_{opt}"
        if key in sample:
            prompt_parts.append(f"  {opt}. {sample[key]}")

    if multi_image:
        prompt_parts.append("\nAnswer with only the letter (A, B, C, or D).")
    else:
        prompt_parts.append("\nAnswer with only the letter (A, B, C, or D).")
    return "\n".join(prompt_parts)


# ==============================================
# 推理
# ==============================================
def run_inference(model, processor, image, prompt, max_new_tokens=2000):
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    generation = generation[0][input_len:]
    return processor.decode(generation, skip_special_tokens=True)


# ==============================================
# 答案提取
# ==============================================
def extract_answer_letter(s: str) -> Optional[str]:
    match = re.search(r'\{([A-Za-z])\}', s)
    if match: return match.group(1).upper()
    match = re.search(r'[Ff]inal [Aa]nswer:\s*([A-Za-z])\b', s)
    if match: return match.group(1).upper()
    match = re.search(r"answer is[:\s]*\n*\s*([A-Z])", s)
    if match: return match.group(1).upper()
    match = re.search(r'\*\*([A-Za-z])\b', s)
    if match: return match.group(1).upper()
    match = re.search(r'(?:^|\n)\s*([A-Za-z])\s*(?:\n|$)', s)
    if match: return match.group(1).upper()
    match = re.search(r'[Tt]he answer is\s*([A-Za-z])\b', s)
    if match: return match.group(1).upper()
    return None


# ==============================================
# 数据集评估（核心逻辑）
# ==============================================
def evaluate_dataset(
    dataset_path, video_base_path, model, processor, output_path,
    include_report=False, num_frames=1, frame_strategy="middle",
    max_samples=None, multi_image=False
):
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    if isinstance(dataset, dict):
        for k in ["data", "samples", "test", "items"]:
            if k in dataset:
                dataset = dataset[k]
                break
    if max_samples:
        dataset = dataset[:max_samples]

    results = []
    correct = total = skipped = 0

    for sample in tqdm(dataset, desc="Evaluating"):
        res = {
            "messages_id": sample.get("messages_id"),
            "question": sample.get("question"),
            "correct_option": sample.get("correct_option"),
        }
        try:
            videos = sample.get("videos", [])
            if not videos:
                res["status"] = "no_video"
                skipped += 1
                results.append(res)
                continue

            if not multi_image:
                vp = os.path.join(video_base_path, videos[0])
                if not os.path.exists(vp):
                    res["status"] = "video_not_found"
                    skipped += 1
                    results.append(res)
                    continue
                frames = extract_frames_from_video(vp, num_frames, frame_strategy)
            else:
                frames = extract_and_concat_frames([os.path.join(video_base_path, v) for v in videos], frame_strategy)

            if not frames:
                res["status"] = "no_frames"
                skipped += 1
                results.append(res)
                continue

            img = frames[0]
            prompt = format_multiple_choice_prompt(sample, include_report, multi_image)
            resp = run_inference(model, processor, img, prompt)
            pred = extract_answer_letter(resp)

            res["model_response"] = resp
            res["predicted_option"] = pred
            res["status"] = "success"
            res["prompt"] = prompt

            if pred and sample.get("correct_option"):
                total += 1
                if pred == sample["correct_option"]:
                    correct += 1
                    res["is_correct"] = True
                else:
                    res["is_correct"] = False

        except Exception as e:
            res["status"] = "error"
            res["error"] = str(e)
            skipped += 1
        results.append(res)

    metrics = {
        "total": len(dataset), "processed": total, "skipped": skipped,
        "correct": correct, "accuracy": correct / total if total else 0
    }
    final = {"metrics": metrics, "results": results}
    os.makedirs(Path(output_path).parent, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(final, f, indent=2)

    print("\n" + "="*50)
    print(f"✅ Evaluation done")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Results saved to: {output_path}")
    return metrics, results


# ==============================================
# 框架主入口（Click + 统一格式）
# ==============================================
@click.command("visual_question_answering")
@click.option("--dataset", type=str, required=True, help="JSON dataset path")
@click.option("--video-base-path", type=str, required=True, help="Video folder")
@click.option("--output", type=str, required=True, help="Output JSON path")
@click.option("--model-id", type=str, default="google/medgemma-1.5-4b-it")
@click.option("--include-report", is_flag=True, default=False)
@click.option("--num-frames", type=int, default=1)
@click.option("--frame-strategy", type=str, default="middle")
@click.option("--max-samples", type=int, default=None)
@click.option("--multi-image", is_flag=True, default=False)
def run(
    dataset, video_base_path, output, model_id,
    include_report, num_frames, frame_strategy, max_samples, multi_image
):
    model, processor = load_model(model_id)
    evaluate_dataset(
        dataset_path=dataset,
        video_base_path=video_base_path,
        model=model,
        processor=processor,
        output_path=output,
        include_report=include_report,
        num_frames=num_frames,
        frame_strategy=frame_strategy,
        max_samples=max_samples,
        multi_image=multi_image
    )


# ==============================================
# 引擎注册（和你项目完全兼容）
# ==============================================
def register():
    return {
        "name": "visual_question_answering",
        "entry": run,
        "description": "MedGemma 1.5 超声视频多选题评估"
    }


if __name__ == "__main__":
    run()