import torch
import json
import os
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

    # save CSV experiment log
    BASE_DIR = os.path.abspath("..")
    exp_dir = os.path.join(BASE_DIR, "outputs/metrics")

    log_experiment(metrics, model_name, exp_dir)

    return metrics
