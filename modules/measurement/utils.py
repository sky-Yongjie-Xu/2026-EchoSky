"""
Purpose:
    Shared utility functions for 2D B-mode and Doppler inference scripts.
    Provides DICOM tag parsing, YBR-to-RGB colour conversion, segmentation
    post-processing, video annotation, and cardiac-phase estimation helpers.

Usage:
    from utils import (
        segmentation_to_coordinates,
        process_video_with_diameter,
        get_coordinates_from_dicom,
        calculate_weighted_centroids_with_meshgrid,
        ybr_to_rgb,
    )
"""
import numpy as np
import cv2
import torch
from typing import Tuple, Union, List
import math
from pathlib import Path
import pydicom
import scipy.signal
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import torchvision

from scipy.signal import savgol_filter

import numpy as np
from scipy.signal import find_peaks


ULTRASOUND_REGIONS_TAG = (0x0018, 0x6011)
REGION_X0_SUBTAG = (0x0018, 0x6018)
REGION_Y0_SUBTAG = (0x0018, 0x601A)
REGION_X1_SUBTAG = (0x0018, 0x601C)
REGION_Y1_SUBTAG = (0x0018, 0x601E)
STUDY_DESCRIPTION_TAG = (0x0008, 0x1030)
SERIES_DESCRIPTION_TAG = (0x0008, 0x103E)
PHOTOMETRIC_INTERPRETATION_TAG = (0x0028, 0x0004)
REGION_PHYSICAL_DELTA_X_SUBTAG = (0x0018, 0x602C)
REGION_PHYSICAL_DELTA_Y_SUBTAG = (0x0018, 0x602E)


def segmentation_to_coordinates(
    logits, normalize=True, order="YX"
):
    predictions_rows, predictions_cols = torch.meshgrid(
        torch.arange(logits.shape[-2], device=logits.device),
        torch.arange(logits.shape[-1], device=logits.device),
        indexing="ij",
    )
    predictions_rows = predictions_rows * logits
    predictions_cols = predictions_cols * logits

    predictions_rows = predictions_rows.sum(dim=(-2, -1)) / (
        logits.sum(dim=(-2, -1)) + 1e-8
    )
    predictions_cols = predictions_cols.sum(dim=(-2, -1)) / (
        logits.sum(dim=(-2, -1)) + 1e-8
    )

    if normalize:
        predictions_rows = predictions_rows / (logits.shape[-2])
        predictions_cols = predictions_cols / (logits.shape[-1])

    if order == "YX":
        predictions = torch.stack([predictions_rows, predictions_cols], dim=-1)
    elif order == "XY":
        predictions = torch.stack([predictions_cols, predictions_rows], dim=-1)
    else:
        raise ValueError("Invalid order")

    return predictions


def get_coordinates_from_dicom(dicom):
    REGION_COORD_SUBTAGS = [
        REGION_X0_SUBTAG,
        REGION_Y0_SUBTAG,
        REGION_X1_SUBTAG,
        REGION_Y1_SUBTAG,
    ]

    if ULTRASOUND_REGIONS_TAG in dicom:
        all_regions = dicom[ULTRASOUND_REGIONS_TAG].value
        regions_with_coords = []
        for region in all_regions:
            region_coords = []
            for coord_subtag in REGION_COORD_SUBTAGS:
                if coord_subtag in region:
                    region_coords.append(region[coord_subtag].value)
                else:
                    region_coords.append(None)

            if all([c is not None for c in region_coords]):
                regions_with_coords.append((region, region_coords))

        regions_with_coords = list(
            sorted(regions_with_coords, key=lambda x: x[1][1], reverse=True)
        )

        return regions_with_coords[0]

    else:
        print("No ultrasound regions found in DICOM file.")
        return None, None


lut = np.load(Path(__file__).parent / "weights/ybr_to_rgb_lut.npy")
def ybr_to_rgb(x):
    return lut[x[..., 0], x[..., 1], x[..., 2]]


def calculate_weighted_centroids_with_meshgrid(logits):
    logits = (logits / logits.max()) * 255
    logits = logits.astype(np.uint8)
    _, binary_image = cv2.threshold(logits, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    centroids = []
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        mask = np.zeros_like(binary_image)
        cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
        h, w = mask.shape
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        mask_indices = mask == 255
        filtered_logits = logits[mask_indices]
        x_coords_filtered = x_coords[mask_indices]
        y_coords_filtered = y_coords[mask_indices]
        weight_sum = filtered_logits.sum()
        if weight_sum != 0:
            cx = (x_coords_filtered * filtered_logits).sum() / weight_sum
            cy = (y_coords_filtered * filtered_logits).sum() / weight_sum
            centroids.append((int(cx), int(cy)))
    centroids = [(int(x), int(y)) for x, y in centroids]
    return centroids, binary_image


def apply_lpf(signal, cutoff):
    fft = np.fft.fft(signal)
    fft[cutoff+1:-cutoff] = 0
    filtered = np.real(np.fft.ifft(fft))
    return filtered


def bpm_to_frame_freq(window_len, fps, bpm):
    beats_per_second_max = bpm / 60
    beats_per_frame_max = beats_per_second_max / fps
    beats_per_video_max = beats_per_frame_max * window_len
    return int(np.ceil(beats_per_video_max))


def process_video_with_diameter(video_path,
                                output_path,
                                df,
                                conversion_factor_X,
                                conversion_factor_Y,
                                ratio,
                                systole_diastole_analysis=False):

    if video_path.endswith(".avi"):
        input_type = "avi"
    elif video_path.endswith(".mp4"):
        input_type = "mp4"

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    frames_array = np.array(frames)

    x1 = df["pred_x1"].values
    y1 = df["pred_y1"].values
    x2 = df["pred_x2"].values
    y2 = df["pred_y2"].values

    delta_x = abs(x2 - x1) * ratio
    delta_y = abs(y2 - y1) * ratio
    diameters = np.sqrt((delta_x * conversion_factor_X)**2 + (delta_y * conversion_factor_Y)**2)

    fps = 30
    cutoff = bpm_to_frame_freq(window_len=len(diameters), fps=fps, bpm=140)
    smooth_diameters = apply_lpf(diameters, cutoff)

    height, width = frames_array[0].shape[:2]
    plot_height = int(width * 0.3)
    output_height = height + plot_height
    output_width = width

    if input_type == "avi":
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (output_width, output_height))
    elif input_type == "mp4":
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (output_width, output_height))

    if systole_diastole_analysis:
        systolic_i, diastolic_i = get_systole_diastole(smooth_diameters,
                                                    smoothing=False,
                                                    kernel=[1, 2, 3, 2, 1],
                                                    distance=25)

        if systolic_i is None or diastolic_i is None:
            return print("No systolic or diastolic peaks found.")

        systolic_diamter = smooth_diameters[systolic_i]
        diastolic_diamter = smooth_diameters[diastolic_i]

        INDEX = 0
        systolic_frame = systolic_i[INDEX] if isinstance(systolic_i, np.ndarray) else systolic_i
        diastolic_frame = diastolic_i[INDEX] if isinstance(diastolic_i, np.ndarray) else diastolic_i
        systolic_diamter = systolic_diamter[INDEX] if isinstance(systolic_diamter, np.ndarray) else systolic_diamter
        diastolic_diamter = diastolic_diamter[INDEX] if isinstance(diastolic_diamter, np.ndarray) else diastolic_diamter
        LVEF_by_teicholz = calculate_lvef_teicholz(diastolic_diameter=diastolic_diamter,
                                                systolic_diameter=systolic_diamter)
        print(f"LVEF by teicholz methods was {LVEF_by_teicholz:.3f} %")

    for i, frame in enumerate(tqdm(frames_array)):
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.plot(smooth_diameters, color='skyblue')
        ax.axvline(x=i, color='black', linestyle='--', alpha=0.5)

        if systole_diastole_analysis:
            ax.scatter(systolic_frame, systolic_diamter, color='blue', marker='o')
            ax.scatter(diastolic_frame, diastolic_diamter, color='red', marker='o')

        ax.set_ylim(0, max(diameters) * 1.1)
        ax.set_xlabel('')
        ax.set_ylabel('Diameter')

        canvas = FigureCanvas(fig)
        canvas.draw()
        try:
            plot_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
            plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        except Exception as e:
            buf = canvas.buffer_rgba()
            plot_image_rgba = np.asarray(buf)
            plot_image = plot_image_rgba[:, :, :3]

        plt.close(fig)
        plot_image = cv2.resize(plot_image, (width, plot_height))
        combined_image = np.vstack((frame, plot_image))

        out.write(cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))

    out.release()

    df["diameter"] = diameters
    df["smooth_diameter"] = smooth_diameters

    if input_type == "avi":
        output_path_replaced = output_path.replace(".avi", ".csv")
    elif input_type == "mp4":
        output_path_replaced = output_path.replace(".mp4", ".csv")

    df.to_csv(output_path_replaced, index=False)
    print(f"Output saved to {output_path} and {output_path_replaced}")


def process_video_with_diameter_tv(video_path,
                                output_path,
                                df,
                                conversion_factor_X,
                                conversion_factor_Y,
                                ratio,
                                systole_diastole_analysis=False):

    cap = cv2.VideoCapture(video_path)
    frames_bgr = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_bgr.append(frame)
    cap.release()

    if not frames_bgr:
        print(f"Error: No frames read from video_path: {video_path}")
        return

    x1_preds = df["pred_x1"].values
    y1_preds = df["pred_y1"].values
    x2_preds = df["pred_x2"].values
    y2_preds = df["pred_y2"].values

    delta_x_diam = abs(x2_preds - x1_preds)
    delta_y_diam = abs(y2_preds - y1_preds)
    diameters = np.sqrt((delta_x_diam * conversion_factor_X)**2 + (delta_y_diam * conversion_factor_Y)**2)

    fps = 30
    cutoff = bpm_to_frame_freq(window_len=len(diameters), fps=fps, bpm=140)
    smooth_diameters = apply_lpf(diameters, cutoff)

    height, width = frames_bgr[0].shape[:2]
    plot_height = int(width * 0.3)
    output_height = height + plot_height
    output_width = width

    if systole_diastole_analysis:
        systolic_i, diastolic_i = get_systole_diastole(smooth_diameters, smoothing=False, kernel=[1, 2, 3, 2, 1], distance=25)

        if systolic_i is None or diastolic_i is None or len(systolic_i)==0 or len(diastolic_i)==0:
            print("No systolic or diastolic peaks found.")
            systolic_frame, diastolic_frame, systolic_diamter, diastolic_diamter, LVEF_by_teicholz = None, None, None, None, None
        else:
            systolic_diamter = smooth_diameters[systolic_i]
            diastolic_diamter = smooth_diameters[diastolic_i]

            INDEX = 0
            systolic_frame = systolic_i[INDEX] if isinstance(systolic_i, np.ndarray) and len(systolic_i) > INDEX else None
            diastolic_frame = diastolic_i[INDEX] if isinstance(diastolic_i, np.ndarray) and len(diastolic_i) > INDEX else None
            systolic_diamter = systolic_diamter[INDEX] if isinstance(systolic_diamter, np.ndarray) and len(systolic_diamter) > INDEX else None
            diastolic_diamter = diastolic_diamter[INDEX] if isinstance(diastolic_diamter, np.ndarray) and len(diastolic_diamter) > INDEX else None

            if systolic_diamter is not None and diastolic_diamter is not None and diastolic_diamter > 0:
                LVEF_by_teicholz = calculate_lvef_teicholz(diastolic_diameter=diastolic_diamter,
                                                        systolic_diameter=systolic_diamter)
            else:
                LVEF_by_teicholz = None
            if LVEF_by_teicholz is not None:
                print(f"LVEF by teicholz methods was {LVEF_by_teicholz:.3f} %")
            else:
                print("LVEF could not be calculated.")
    else:
        systolic_frame, diastolic_frame, systolic_diamter, diastolic_diamter, LVEF_by_teicholz = None, None, None, None, None

    final_video_frames_tensors = []
    for i, frame_bgr in enumerate(tqdm(frames_bgr)):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(figsize=(8, 2))
        ax.plot(diameters, label='Raw Diameter', alpha=0.6)
        ax.plot(smooth_diameters, label='Smoothed Diameter', color='red')
        ax.axvline(x=i, color='black', linestyle='--', alpha=0.5)

        if systole_diastole_analysis and systolic_frame is not None and diastolic_frame is not None:
            if isinstance(systolic_i, np.ndarray) and len(systolic_i) > 0:
                ax.scatter(systolic_i, smooth_diameters[systolic_i], color='blue', marker='o', label='Systole Peaks')
            if isinstance(diastolic_i, np.ndarray) and len(diastolic_i) > 0:
                ax.scatter(diastolic_i, smooth_diameters[diastolic_i], color='yellow', marker='o', label='Diastole Peaks')
            ax.legend()

        ax.set_ylim(0, max(diameters) * 1.1 if len(diameters) > 0 else 1.0)
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Diameter (cm)')
        ax.set_title(f'Diameter Over Time (LVEF: {LVEF_by_teicholz:.2f}%)' if LVEF_by_teicholz is not None else 'Diameter Over Time')

        canvas = FigureCanvas(fig)
        canvas.draw()
        try:
            plot_image_rgb = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
            plot_image_rgb = plot_image_rgb.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        except Exception as e:
            buf = canvas.buffer_rgba()
            plot_image_rgba = np.asarray(buf)
            plot_image_rgb = plot_image_rgba[:, :, :3]

        plt.close(fig)
        plot_image_rgb = cv2.resize(plot_image_rgb, (width, plot_height))
        combined_image_rgb = np.vstack((frame_rgb, plot_image_rgb))
        final_video_frames_tensors.append(torch.from_numpy(combined_image_rgb))

    if final_video_frames_tensors:
        video_tensor_to_write = torch.stack(final_video_frames_tensors)
        torchvision.io.write_video(
            filename=output_path,
            video_array=video_tensor_to_write,
            fps=fps,
            video_codec='libx264'
        )
    else:
        print(f"Warning: No frames to write for output video: {output_path}")

    df["diameter"] = diameters
    df["smooth_diameter"] = smooth_diameters
    df.to_csv(output_path.replace(".mp4", ".csv"), index=False)

    if systole_diastole_analysis:
        return LVEF_by_teicholz
    return None


def get_systole_diastole(diameter,
                         smoothing=False,
                         kernel=[1, 2, 3, 2, 1],
                         distance=25):

    if smoothing:
        kernel = np.array(kernel)
        kernel = kernel / kernel.sum()
        diameter = np.convolve(diameter, kernel, mode='same')

    diastole_i, _ = find_peaks(diameter, distance=distance)
    systole_i, _ = find_peaks(-diameter, distance=distance)

    if len(systole_i) != 0 and len(diastole_i) != 0:
        start_minmax = np.concatenate([diastole_i, systole_i]).min()
        end_minmax = np.concatenate([diastole_i, systole_i]).max()
        diastole_i = np.delete(diastole_i, np.where((diastole_i == start_minmax) | (diastole_i == end_minmax)))
        systole_i = np.delete(systole_i, np.where((systole_i == start_minmax) | (systole_i == end_minmax)))

    return systole_i, diastole_i


def calculate_lvef_teicholz(diastolic_diameter,
                            systolic_diameter):
    if systolic_diameter > diastolic_diameter:
        print("Error: Systolic diameter cannot be greater than diastolic diameter.")
        return None

    lvedv = (7.0 / (2.4 + diastolic_diameter)) * (diastolic_diameter ** 3)
    lvesv = (7.0 / (2.4 + systolic_diameter)) * (systolic_diameter ** 3)
    sv = lvedv - lvesv
    lvef = (sv / lvedv) * 100

    return lvef


def process_diameter(df,
                    conversion_factor_X,
                    conversion_factor_Y
                    ):
    x1 = df["pred_x1"].values
    y1 = df["pred_y1"].values
    x2 = df["pred_x2"].values
    y2 = df["pred_y2"].values

    delta_x = x2 - x1
    delta_y = y2 - y1
    raw_diameters = np.sqrt((delta_x * conversion_factor_X)**2 + (delta_y * conversion_factor_Y)**2)

    fps = 30
    cutoff = bpm_to_frame_freq(window_len=len(raw_diameters), fps=fps, bpm=140)
    smooth_diameters = apply_lpf(raw_diameters, cutoff)

    return raw_diameters, smooth_diameters