import torch
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from utils.metrics import compute_metrics
from utils.experiment_logger import log_experiment

def evaluate(model, dataloader, device, save_path=None, model_name="unknown"):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    metrics = compute_metrics(y_true, y_pred)

    print("Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # save json
    if save_path:
        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=4)

    # 🔥 save CSV experiment log
    BASE_DIR = os.path.abspath("..")
    exp_dir = os.path.join(BASE_DIR, "outputs/metrics")

    log_experiment(metrics, model_name, exp_dir)

    return metrics

def plot_confusion_matrix(y_true, y_pred, class_names, figsize=(10,8)):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    return plt

def visualize_misclassified_samples(model, dataloader, device, class_names, max_samples=5):
    """Visualize TP, FP, FN for each class"""
    model.eval()
    tp_imgs, fp_imgs, fn_imgs = [], [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)

            for i in range(len(y)):
                img_np = x[i].cpu().permute(1,2,0).numpy()  # assuming image [C,H,W]
                label = y[i].item()
                pred = preds[i].item()
                if label == pred and len(tp_imgs) < max_samples:
                    tp_imgs.append((img_np, label, pred))
                elif label != pred:
                    if len(fp_imgs) < max_samples:
                        fp_imgs.append((img_np, label, pred))
                    if len(fn_imgs) < max_samples:
                        fn_imgs.append((img_np, label, pred))

    return tp_imgs, fp_imgs, fn_imgs

def create_visualization_figure(tp_imgs, fp_imgs, fn_imgs, class_names):
    """Combine TP, FP, FN into one figure"""
    fig, axes = plt.subplots(3, max(len(tp_imgs), len(fp_imgs), len(fn_imgs)), figsize=(15,9))
    categories = ['TP', 'FP', 'FN']
    imgs_lists = [tp_imgs, fp_imgs, fn_imgs]

    for row, imgs in enumerate(imgs_lists):
        for col in range(len(imgs)):
            axes[row, col].imshow(imgs[col][0])
            axes[row, col].axis('off')
            axes[row, col].set_title(f"GT:{class_names[imgs[col][1]]}\nPred:{class_names[imgs[col][2]]}")
        # hide empty axes
        for col in range(len(imgs), axes.shape[1]):
            axes[row, col].axis('off')

    for ax, cat in zip(axes[:,0], categories):
        ax.set_ylabel(cat, fontsize=14)
    plt.tight_layout()
    return fig

def evaluate_with_visualization(model, dataloader, device, class_names, save_path=None, model_name="unknown", max_samples=5):
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    metrics = compute_metrics(y_true, y_pred)

    print("Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # save json
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=4)

    # 🔥 save CSV experiment log
    BASE_DIR = os.path.abspath("..")
    exp_dir = os.path.join(BASE_DIR, "outputs/metrics")
    log_experiment(metrics, model_name, exp_dir)

    # --- visualization ---
    tp_imgs, fp_imgs, fn_imgs = visualize_misclassified_samples(model, dataloader, device, class_names, max_samples)
    fig = create_visualization_figure(tp_imgs, fp_imgs, fn_imgs, class_names)

    # save figure
    if save_path:
        fig_path = save_path.replace(".json", "_visualization.png")
        fig.savefig(fig_path)
        plt.close(fig)

    # return metrics + figure
    return metrics, fig