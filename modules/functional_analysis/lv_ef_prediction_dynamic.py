"""Functions for training and running EF prediction."""

import math
import os
import time
import gc  # 关键：垃圾回收

import click
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import torch
import torchvision
import tqdm

import echonet


@click.command("lv_ef_prediction_dynamic")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default=None)
@click.option("--output", type=click.Path(file_okay=False), default=None)
@click.option("--task", type=str, default="EF")
@click.option("--model_name", type=click.Choice(
    sorted(name for name in torchvision.models.video.__dict__
           if name.islower() and not name.startswith("__") and callable(torchvision.models.video.__dict__[name]))),
    default="r2plus1d_18")
@click.option("--pretrained/--random", default=True)
@click.option("--weights", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--run_test/--skip_test", default=False)
@click.option("--num_epochs", type=int, default=45)
@click.option("--lr", type=float, default=1e-4)
@click.option("--weight_decay", type=float, default=1e-4)
@click.option("--lr_step_period", type=int, default=15)
@click.option("--frames", type=int, default=32)
@click.option("--period", type=int, default=2)
@click.option("--num_train_patients", type=int, default=None)
@click.option("--num_workers", type=int, default=4)
@click.option("--batch_size", type=int, default=20)
@click.option("--device", type=str, default=None)
@click.option("--seed", type=int, default=0)
def run(
    data_dir=None,
    output=None,
    task="EF",

    model_name="r2plus1d_18",
    pretrained=True,
    weights=None,

    run_test=False,
    num_epochs=45,
    lr=1e-4,
    weight_decay=1e-4,
    lr_step_period=15,
    frames=32,
    period=2,
    num_train_patients=None,
    num_workers=4,
    batch_size=20,
    device=None,
    seed=0,
):
    """Trains/tests EF prediction model.
    """

    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set default output directory
    if output is None:
        output = os.path.join("output", "video", "{}_{}_{}_{}".format(model_name, frames, period, "pretrained" if pretrained else "random"))
    os.makedirs(output, exist_ok=True)

    # Set device for computations
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = torchvision.models.video.__dict__[model_name](pretrained=pretrained)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.fc.bias.data[0] = 55.6
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    if weights is not None:
        checkpoint = torch.load(weights)
        model.load_state_dict(checkpoint['state_dict'])

    # Set up optimizer
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

    # Compute mean and std
    mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(root=data_dir, split="train"))
    kwargs = {"target_type": task,
              "mean": mean,
              "std": std,
              "length": frames,
              "period": period,
              }

    # Set up datasets and dataloaders
    dataset = {}
    dataset["train"] = echonet.datasets.Echo(root=data_dir, split="train", **kwargs, pad=12)
    if num_train_patients is not None and len(dataset["train"]) > num_train_patients:
        indices = np.random.choice(len(dataset["train"]), num_train_patients, replace=False)
        dataset["train"] = torch.utils.data.Subset(dataset["train"], indices)
    dataset["val"] = echonet.datasets.Echo(root=data_dir, split="val", **kwargs)

    # Run training and testing loops
    with open(os.path.join(output, "log.csv"), "a") as f:
        epoch_resume = 0
        bestLoss = float("inf")
        try:
            checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['opt_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_dict'])
            epoch_resume = checkpoint["epoch"] + 1
            bestLoss = checkpoint["best_loss"]
            f.write("Resuming from epoch {}\n".format(epoch_resume))
        except FileNotFoundError:
            f.write("Starting run from scratch\n")

        for epoch in range(epoch_resume, num_epochs):
            print("Epoch #{}".format(epoch), flush=True)
            for phase in ['train', 'val']:
                start_time = time.time()
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)

                ds = dataset[phase]
                dataloader = torch.utils.data.DataLoader(
                    ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=(phase == "train"))

                loss, yhat, y = echonet.utils.video.run_epoch(model, dataloader, phase == "train", optim, device)
                f.write("{},{},{},{},{},{},{},{},{}\n".format(epoch,
                                                              phase,
                                                              loss,
                                                              sklearn.metrics.r2_score(y, yhat),
                                                              time.time() - start_time,
                                                              y.size,
                                                              sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                                              sum(torch.cuda.max_memory_reserved() for i in range(torch.cuda.device_count())),
                                                              batch_size))
                f.flush()

                # 清缓存
                del loss, yhat, y, dataloader
                gc.collect()
                torch.cuda.empty_cache()

            scheduler.step()

            # Save checkpoint
            save = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'period': period,
                'frames': frames,
                'best_loss': bestLoss,
                'loss': loss,
                'r2': sklearn.metrics.r2_score(y, yhat),
                'opt_dict': optim.state_dict(),
                'scheduler_dict': scheduler.state_dict(),
            }
            torch.save(save, os.path.join(output, "checkpoint.pt"))
            if loss < bestLoss:
                torch.save(save, os.path.join(output, "best.pt"))
                bestLoss = loss

        # Load best weights
        if num_epochs != 0:
            checkpoint = torch.load(os.path.join(output, "best.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            f.write("Best validation loss {} from epoch {}\n".format(checkpoint["loss"], checkpoint["epoch"]))
            f.flush()

        # ===================== 测试阶段：支持 all clips，永不爆内存 =====================
        if run_test:
            for split in ["val", "test"]:
                print(f"\n=== Running {split} (one clip) ===")

                # -------------------------- one clip --------------------------
                dataloader = torch.utils.data.DataLoader(
                    echonet.datasets.Echo(root=data_dir, split=split, **kwargs),
                    batch_size=batch_size, num_workers=0, shuffle=False, pin_memory=False)
                loss, yhat, y = echonet.utils.video.run_epoch(model, dataloader, False, None, device)

                f.write("{} (one clip) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(split, *echonet.utils.bootstrap(y, yhat, sklearn.metrics.r2_score)))
                f.write("{} (one clip) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(split, *echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_absolute_error)))
                f.write("{} (one clip) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, *tuple(map(math.sqrt, echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_squared_error)))))
                f.flush()

                # 画图
                echonet.utils.latexify()
                fig = plt.figure(figsize=(3, 3))
                lower = min(y.min(), yhat.min())
                upper = max(y.max(), yhat.max())
                plt.scatter(y, yhat, color="k", s=1, edgecolor=None, zorder=2)
                plt.plot([0, 100], [0, 100], linewidth=1, zorder=3)
                plt.axis([lower - 3, upper + 3, lower - 3, upper + 3])
                plt.gca().set_aspect("equal", "box")
                plt.xlabel("Actual EF (%)")
                plt.ylabel("Predicted EF (%)")
                plt.xticks([10,20,30,40,50,60,70,80])
                plt.yticks([10,20,30,40,50,60,70,80])
                plt.grid(color="gainsboro", linestyle="--", linewidth=1, zorder=1)
                plt.tight_layout()
                plt.savefig(os.path.join(output, f"{split}_scatter.pdf"))
                plt.close(fig)

                fig = plt.figure(figsize=(3, 3))
                plt.plot([0,1], [0,1], linewidth=1, color="k", linestyle="--")
                for thresh in [35,40,45,50]:
                    fpr, tpr, _ = sklearn.metrics.roc_curve(y > thresh, yhat)
                    plt.plot(fpr, tpr)
                plt.axis([-0.01,1.01,-0.01,1.01])
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.tight_layout()
                plt.savefig(os.path.join(output, f"{split}_roc.pdf"))
                plt.close(fig)

                # 清空所有 one clip 变量
                del dataloader, loss, yhat, y, fig
                gc.collect()
                torch.cuda.empty_cache()

                # -------------------------- all clips（能跑！） --------------------------
                print(f"=== Running {split} (all clips) ===")
                ds = echonet.datasets.Echo(root=data_dir, split=split, **kwargs, clips="all")
                dataloader = torch.utils.data.DataLoader(
                    ds, batch_size=1, num_workers=0, shuffle=False, pin_memory=False)

                model.eval()
                y_true = []
                y_pred_mean = []

                # 逐视频写入文件，不存内存
                with open(os.path.join(output, f"{split}_predictions.csv"), "w") as g:
                    with torch.no_grad():
                        for idx, (X, ef) in enumerate(tqdm.tqdm(dataloader, desc=f"{split} all clips")):
                            fname = ds.fnames[idx]
                            y_true.append(ef.item())

                            # 送到设备
                            X = X.to(device)
                            b, nc, c, f, h, w = X.shape
                            X = X.view(-1, c, f, h, w)

                            # 逐块推理，block_size=1 最省内存
                            preds = []
                            for j in range(0, X.shape[0], 1):
                                batch_x = X[j:j+1]
                                out = model(batch_x)
                                preds.append(out.detach().cpu().numpy())

                                # 立即写入！不存内存
                                g.write(f"{fname},{j},{out.item():.4f}\n")
                                g.flush()

                                # 删！删！删！
                                del batch_x, out
                                gc.collect()
                                torch.cuda.empty_cache()

                            # 计算均值
                            preds = np.concatenate(preds).mean()
                            y_pred_mean.append(preds)

                            # 删视频数据
                            del X, ef
                            gc.collect()
                            torch.cuda.empty_cache()

                # 计算指标
                y_true = np.array(y_true)
                y_pred_mean = np.array(y_pred_mean)

                f.write("{} (all clips) R2:  {:.3f}\n".format(split, sklearn.metrics.r2_score(y_true, y_pred_mean)))
                f.write("{} (all clips) MAE: {:.2f}\n".format(split, sklearn.metrics.mean_absolute_error(y_true, y_pred_mean)))
                f.flush()

                # 清空 all clips 变量
                del ds, dataloader, y_true, y_pred_mean
                gc.collect()
                torch.cuda.empty_cache()


def run_epoch(model, dataloader, train, optim, device, save_all=False, block_size=None):
    model.train(train)
    total = 0
    n = 0
    yhat = []
    y = []

    with torch.set_grad_enabled(train):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for X, outcome in dataloader:
                y.append(outcome.numpy())
                X = X.to(device)
                outcome = outcome.to(device)

                average = (len(X.shape) == 6)
                if average:
                    X = X.view(-1, *X.shape[2:])

                if block_size is None:
                    outputs = model(X)
                else:
                    outputs = torch.cat([model(X[j:j+block_size]) for j in range(0, len(X), block_size)])

                if average:
                    outputs = outputs.view(outcome.shape[0], -1).mean(1)

                yhat.append(outputs.detach().cpu().numpy())
                loss = torch.nn.functional.mse_loss(outputs, outcome)

                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                total += loss.item() * len(X)
                n += len(X)
                pbar.set_postfix(loss=f"{total/n:.2f}")
                pbar.update()

                # 每步清
                del X, outcome, outputs, loss
                gc.collect()
                if train:
                    torch.cuda.empty_cache()

    yhat = np.concatenate(yhat)
    y = np.concatenate(y)
    return total / n, yhat, y

def register():
    """
    模块注册接口：用于主引擎动态加载
    你的主程序可以通过这个函数自动识别模块、调用run()
    """
    return {
        "name": "lv_ef_prediction_dynamic",
        "entry": run,
        "description": "EF 射血分数预测"
    }

if __name__ == "__main__":
    run()