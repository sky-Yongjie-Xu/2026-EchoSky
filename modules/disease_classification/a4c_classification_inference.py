from argparse import ArgumentParser
from operator import index, mod
from pandas.core.algorithms import isin
import torch
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Union
import click  # 加入 click
from typing import Iterable, Tuple
from scipy.signal import find_peaks
import cv2
import argparse
import torchvision


# relative paths to weights for various models
weights_path = Path(__file__).parent / 'weights'
model_paths = {
    'plax': weights_path / 'hypertrophy_model.pt',
    'amyloid': weights_path / 'amyloid.pt',
    'as': weights_path / 'as_model.pt'
}


class Model(torch.nn.Module):

    """Binary video classification used to classify heart conditions.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = torchvision.models.video.r3d_18()
        self.model.fc = torch.nn.Linear(in_features=512, out_features=1, bias=True)
    
    def forward(self, x):
        return self.model(x)
    

class BoolAction(argparse.Action):

    """Class used by argparse to parse binary arguements.
    Yes, Y, y, True, T, t are all accepted as True. Any other
    arguement is evaluated as False.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        b = values.lower()[0] in ['t', 'y', '1']
        setattr(namespace, self.dest, b)
        print(parser)


def get_clip_dims(paths: Iterable[Union[Path, str]]) -> Tuple[np.ndarray, list]:
    """Gets the dimentions of all the videos in a list of paths.

    Args:
        paths (Iterable[Union[Path, str]]): List of paths to iterrate through

    Returns:
        dims (np.ndarray): array of clip dims (frames, width, height). shape=(n, 3)
        filenames (list): list of filenames. len=n
    """
    
    dims = []
    fnames = []
    for p in paths:
        if isinstance(p, str):
            p = Path(p)
        if '.avi' not in p.name:
            continue
        cap = cv2.VideoCapture(str(p))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        dims.append((frame_count, w, h))
        fnames.append(p.name)
    return np.array(dims).T, fnames

def read_clip(path, res=None, max_len=None) -> np.ndarray:
    """Reads a clip and returns it as a numpy array

    Args:
        path ([Path, str]): Path to video to read
        res (Tuple[int], optional): Resolution of video to return. If None, 
            original resolution will be returned otherwise the video will be 
            cropped and downsampled. Defaults to None.
        max_len (int, optional): Max length of video to read. Only the first n 
            frames of longer videos will be returned. Defaults to None.

    Returns:
        np.ndarray: Numpy array of video. shape=(n, h, w, 3)
    """

    cap = cv2.VideoCapture(str(path))
    frames = []
    i = 0
    while True:
        if max_len is not None and i >= max_len:
            break
        i += 1
        ret, frame = cap.read()
        if not ret:
            break
        if res is not None:
            frame = crop_and_scale(frame, res)
        frames.append(frame)
    cap.release()
    return np.array(frames)

def get_systole_diastole(lvid: np.ndarray, kernel=[1, 2, 3, 2, 1], distance: int=25) -> Tuple[np.ndarray]:
    """Finds heart phase from a representative signal. Signal must be maximum at end diastole and
    minimum at end systole.

    Args:
        lvid (np.ndarray): Signal representing heart phase. shape=(n,)
        kernel (list, optional): Smoothing kernel used before finding peaks. Defaults to [1, 2, 3, 2, 1].
        distance (int, optional): Minimum distance between peaks in find_peaks(). Defaults to 25.

    Returns:
        systole_i (np.ndarray): Indices of end systole. shape=(n_sys,)
        diastole_i (np.ndarray): Indices of end diastole. shape=(n_dia,)
    """

    # Smooth input
    kernel = np.array(kernel)
    kernel = kernel / kernel.sum()
    lvid_filt = np.convolve(lvid, kernel, mode='same')

    # Find peaks
    diastole_i, _ = find_peaks(lvid_filt, distance=distance)
    systole_i, _ = find_peaks(-lvid_filt, distance=distance)

    # Ignore first/last index if possible
    if len(systole_i) != 0 and len(diastole_i) != 0:
        start_minmax = np.concatenate([diastole_i, systole_i]).min()
        end_minmax = np.concatenate([diastole_i, systole_i]).max()
        diastole_i = np.delete(diastole_i, np.where((diastole_i == start_minmax) | (diastole_i == end_minmax)))
        systole_i = np.delete(systole_i, np.where((systole_i == start_minmax) | (systole_i == end_minmax)))
    
    return systole_i, diastole_i

def get_lens_np(pts: np.ndarray) -> np.ndarray:
    """Used to get the euclidean distance between consecutive points.

    Args:
        pts (np.ndarray): Input points. shape=(..., n, 2)

    Returns:
        np.ndarray: Distances. shape=(..., n-1)
    """
    return np.sum((pts[..., 1:, :] - pts[..., :-1, :]) ** 2, axis=-1) ** 0.5

def get_points_np(preds: np.ndarray, threshold: float=0.3) -> np.ndarray:
    """Gets the centroid of heatmaps.

    Args:
        preds (np.ndarray): Input heatmaps. shape=(n, h, w, c)
        threshold (float, optional): Value below which input pixels are ignored. Defaults to 0.3.

    Returns:
        np.ndarray: Centroid locations. shape=(n, c, 2)
    """

    preds = np.copy(preds)
    preds[preds < threshold] = 0
    Y, X = np.mgrid[:preds.shape[-3], :preds.shape[-2]]
    np.seterr(divide='ignore', invalid='ignore')
    x_pts = np.sum(X[None, ..., None] * preds, axis=(-3, -2)) / np.sum(preds, axis=(-3, -2))
    y_pts = np.sum(Y[None, ..., None] * preds, axis=(-3, -2)) / np.sum(preds, axis=(-3, -2))
    return np.moveaxis(np.array([x_pts, y_pts]), 0, -1)

def get_angles_np(pts: np.ndarray) -> np.ndarray:
    """Returns the angles between corresponding segments of a polyline.

    Args:
        pts (np.ndarray): Input polyline. shape=(..., n, 2)

    Returns:
        np.ndarray: Angles in degrees. Constrained to [-180, 180]. shape=(..., n-1)
    """

    a_m = np.arctan2(*np.moveaxis(pts[..., 1:, :] - pts[..., :-1, :], -1, 0))
    a = (a_m[..., 1:] - a_m[..., :-1]) * 180 / np.pi
    a[a > 180] -= 360
    a[a < -180] += 360
    return a

def get_pred_measurements(preds: np.ndarray, scale: float=1) -> Tuple[np.ndarray]:
    """Given PLAX heatmap predictions, generate values of interest.

    Args:
        preds (np.ndarray): PLAX model heatmap predictions. shape=(n, h, w, 4)
        scale (int, optional): Image scale [cm/px]. Defaults to 1.

    Returns:
        pred_pts (np.ndarray): Centroids of heatmaps. shape=(n, 4, 2)
        pred_lens (np.ndarray): Measurement lengths. shape=(n, 3)
        sys_i (np.ndarray): Indices of end systole. shape=(n_sys,)
        dia_i (np.ndarray): Indices of end diastole. shape=(n_dia,)
        angles (np.ndarray): Angles between measurements in degrees. shape=(n, 2)
    """

    pred_pts = get_points_np(preds)
    pred_lens = get_lens_np(pred_pts) * scale
    sys_i, dia_i = get_systole_diastole(pred_lens[:, 1])
    angles = get_angles_np(pred_pts)
    return pred_pts, pred_lens, sys_i, dia_i, angles

def overlay_preds(
            a: np.ndarray, 
            background=None, 
            c=np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1], [0, 1, 0]])
        ) -> np.ndarray:
    """Used to visualize PLAX model predictions over echo frames

    Args:
        a (np.ndarray): Predicted heatmaps. shape=(h, w, 4)
        background (np.ndarray, optional): Echo frame to overlay on top of. shape=(h, w, 3) Defaults to None.
        c (np.ndarray, optional): RGB colors corresponding to each channel of the predictions. shape=(4, 3)
            Defaults to np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1], [0, 1, 0]]).

    Returns:
        np.ndarray: RGB image visualization of heatmaps. shape=(h, w, 3)
    """

    if background is None:
        background = np.zeros((a.shape[0], a.shape[1], 3))
    np.seterr(divide='ignore', invalid='ignore')
    color = (a ** 2).dot(c) / np.sum(a, axis=-1)[..., None]
    alpha = (1 - np.prod(1 - a, axis=-1))[..., None]
    alpha = np.nan_to_num(alpha)
    color = np.nan_to_num(color)
    return alpha * color + (1 - alpha) * background

def crop_and_scale(img: np.ndarray, res=(640, 480)) -> np.ndarray:
    """Scales and cropts an numpy array image to specified resolution.
    Image is first cropped to correct aspect ratio and then scaled using
    bicubic interpolation.

    Args:
        img (np.ndarray): Image to be resized. shape=(h, w, 3)
        res (tuple, optional): Resolution to be scaled to. Defaults to (640, 480).

    Returns:
        np.ndarray: Scaled image. shape=(res[1], res[0], 3)
    """

    in_res = (img.shape[1], img.shape[0])
    r_in = in_res[0] / in_res[1]
    r_out = res[0] / res[1]

    if r_in > r_out:
        padding = int(round((in_res[0] - r_out * in_res[1]) / 2))
        img = img[:, padding:-padding]
    if r_in < r_out:
        padding = int(round((in_res[1] - in_res[0] / r_out) / 2))
        img = img[padding:-padding]
    
    img = cv2.resize(img, res)

    return img


class A4cClassificationInferenceEngine:

    def __init__(self, model_path: Union[Path, str]=model_paths['amyloid'], device: str='cuda:0', res=(112, 112)) -> None:
        if isinstance(model_path, str):
            model_path = Path(model_path)
        self.model_path = model_path
        self.device = device
        self.res = res
        self.model = None
    
    def load_model(self):
        self.model = Model()
        self.model.to(self.device)
        return self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))

    def run_on_dir(self, 
                in_dir: Union[Path, str], 
                out_dir: Union[Path, str], 
                batch_size: int=4, clip_length: int=96, 
                verbose: bool=True, threshold: float=0.5
            ) -> None:

        if not isinstance(in_dir, Path):
            in_dir = Path(in_dir)
        if not isinstance(out_dir, Path):
            out_dir = Path(out_dir)
        if not out_dir.exists():
            out_dir.mkdir()
        
        if self.model is None:
            self.load_model()
        
        def batch_gen():
            batch = ([], [])
            for p in tqdm(list(in_dir.iterdir())) if verbose else in_dir.iterdir():
                if '.avi' not in p.name:
                    continue
                clip = read_clip(p, res=self.res, max_len=clip_length)
                if len(clip) != clip_length:
                    continue
                batch[0].append(p)
                batch[1].append(clip)
                if len(batch[0]) == batch_size:
                    yield batch[0], np.array(batch[1])
                    batch = ([], [])
            if len(batch[0]) != 0:
                yield batch[0], np.array(batch[1])
        
        results = {'Filename': [], 'Positive Confidence': []}
        for paths, clips in batch_gen():
            clips = torch.from_numpy(np.moveaxis(np.array(clips), -1, 1)).to(torch.float).to(self.device) / 255.0
            with torch.no_grad():
                preds = torch.sigmoid(self.model(clips)).detach().cpu().numpy()
                results['Filename'].append([p.name for p in paths])
                results['Positive Confidence'].append(preds[:, 0])
        
        results = pd.DataFrame({k: np.concatenate(v) for k, v in results.items()})
        n_pos = (results['Positive Confidence'] > threshold).sum()
        print(f'{n_pos}/{len(results)} ({100 * n_pos / len(results):.2f}%) predicted positive')
        results.to_csv(out_dir / (in_dir.name + '.csv'), index=False)


# ======================== Click 命令封装 ========================
@click.command("amyloid_classification")
@click.option("--in_dir", type=click.Path(exists=True, file_okay=False), required=True, help="输入视频目录")
@click.option("--out_dir", type=click.Path(file_okay=False), required=True, help="输出目录")
@click.option("--device", default="cuda:0", help="运行设备 cuda:0 / cpu")
@click.option("--batch_size", default=8, help="批次大小")
@click.option("--clip_length", default=96, help="推理使用的帧数")
@click.option("--verbose/--quiet", default=True, help="显示进度")
@click.option("--model_path", type=click.Path(), default=model_paths['amyloid'], help="模型权重路径")
@click.option("--threshold", default=0.5, help="阳性判断阈值")
def run(
    in_dir,
    out_dir,
    device="cuda:0",
    batch_size=8,
    clip_length=96,
    verbose=True,
    model_path=model_paths['amyloid'],
    threshold=0.5
):
    engine = A4cClassificationInferenceEngine(model_path=model_path, device=device)
    engine.run_on_dir(
        in_dir=in_dir,
        out_dir=out_dir,
        batch_size=batch_size,
        clip_length=clip_length,
        verbose=verbose,
        threshold=threshold
    )


# ======================== 引擎注册接口 ========================
def register():
    return {
        "name": "a4c_classification",
        "entry": run,
        "description": "淀粉样变分类（A4C 心超二分类推理）"
    }


if __name__ == "__main__":
    run()