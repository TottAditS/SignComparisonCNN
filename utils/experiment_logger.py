import os
import csv
from datetime import datetime
def log_experiment(metrics, model_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    file_path = os.path.join(save_dir, "experiments.csv")

    file_exists = os.path.isfile(file_path)

    data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        **metrics
    }

    with open(file_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)

    print(f"Experiment logged to: {file_path}")