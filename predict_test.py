import argparse
import os
import pickle
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
DEFAULT_DATA_ROOT = os.path.join(ROOT_DIR, "data")
DEFAULT_DATASET = "dualkiba-DualPGdq(work2)"
DEFAULT_MODEL_PATH = "/New_SSD/demo_hpf/MSN-DTA/MSN-DTA/code/save/20260904_160059_dualkiba-DualPGdq(work2random42)/model/Epoch-307, Train_MSE-0.0383, Test_MSE-0.1381, Test_CI-0.8953, Test_RM2-0.7756, Test_Pearson-0.8964.pt"
DEFAULT_OUTPUT_PATH = None
DEFAULT_BATCH_SIZE = 128
DEFAULT_NUM_WORKERS = 0

for path in reversed(
    (
        BASE_DIR,
        os.path.join(ROOT_DIR, "MSN-DTA"),
        os.path.join(ROOT_DIR, "all code"),
    )
):
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

from dataloader import convert_batch_pr
from metrics import get_cindex, get_pearson_np, get_rm2
from work16MSNmodel12 import MSN_DTA
from work16dataset import PreDataset


def collate_fn(data_list):
    batch_pr = convert_batch_pr([data.target for data in data_list])
    data = Batch.from_data_list(data_list)
    data.target = batch_pr["input_ids"]
    data.target_mask = batch_pr["attention_mask"]
    return data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load one saved MSN-DTA .pt file and predict the test set."
    )
    parser.add_argument(
        "--model_path",
        default=DEFAULT_MODEL_PATH,
        help="Path to the .pt file saved by work16main.py.",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Dataset directory name under --data_root.",
    )
    parser.add_argument(
        "--data_root",
        default=DEFAULT_DATA_ROOT,
        help="Directory containing the dataset directory.",
    )
    parser.add_argument(
        "--output_path",
        default=DEFAULT_OUTPUT_PATH,
        help="CSV output path. Defaults to test_predictions.csv beside the .pt file.",
    )
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument(
        "--device",
        default=None,
        help="For example cuda:0 or cpu. Defaults to CUDA when available.",
    )
    return parser.parse_args()


def load_cluster_centers(dataset_root):
    c_path = os.path.join(dataset_root, "processed", "C_1280_2000_8.pkl")
    with open(c_path, "rb") as f:
        cluster_data = pickle.load(f)

    c = torch.as_tensor(
        cluster_data["cluster_centers"], dtype=torch.float32
    ).permute(1, 0)
    text_c = torch.as_tensor(
        cluster_data["text_cluster_centers"], dtype=torch.float32
    )
    return c, text_c


def load_model(model_path, c, text_c, device):
    model = MSN_DTA(
        3,
        22,
        25 + 1,
        embedding_size=128,
        filter_num=32,
        out_dim=1,
        c=c,
        text_c=text_c,
        blpretrain=False,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]

    if not isinstance(checkpoint, dict):
        raise TypeError("The .pt file does not contain a model state_dict.")

    state_dict = {
        key.removeprefix("module."): value for key, value in checkpoint.items()
    }
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict(model, dataloader, device):
    y_trues = []
    y_preds = []

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Predicting"):
            data = data.to(device)
            pred = model(data, 1, 9999)
            y_trues.append(data.y.detach().cpu().numpy().reshape(-1))
            y_preds.append(pred.detach().cpu().numpy().reshape(-1))

    if not y_trues:
        raise RuntimeError("The test set is empty, so no predictions were saved.")

    return np.concatenate(y_trues), np.concatenate(y_preds)


def save_predictions(y_true, y_pred, output_path):
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.DataFrame(
        {
            "sample_index": np.arange(len(y_true)),
            "y_true": y_true,
            "y_pred": y_pred,
            "abs_error": np.abs(y_true - y_pred),
            "squared_error": np.square(y_true - y_pred),
        }
    )
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def main():
    args = parse_args()

    if not args.model_path:
        raise ValueError(
            "Please provide --model_path with the .pt file to evaluate."
        )

    model_path = os.path.abspath(args.model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    data_root = os.path.abspath(args.data_root)
    dataset_root = os.path.join(data_root, args.dataset)
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_root}. "
            "Check --data_root and --dataset."
        )

    output_path = args.output_path
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(model_path), "test_predictions.csv"
        )

    device = torch.device(
        args.device
        if args.device
        else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")
    print(f"Loading model: {model_path}")
    print(f"Dataset: {dataset_root}")

    test_set = PreDataset(dataset_root, train=False)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )

    c, text_c = load_cluster_centers(dataset_root)
    model = load_model(model_path, c, text_c, device)
    y_true, y_pred = predict(model, test_loader, device)
    output_path = save_predictions(y_true, y_pred, output_path)

    mse = float(np.mean(np.square(y_true - y_pred)))
    cindex = get_cindex(y_true, y_pred)
    rm2 = get_rm2(y_true, y_pred)
    pearson = get_pearson_np(y_true, y_pred)

    print(f"Test samples: {len(y_true)}")
    print(f"Saved y_true/y_pred: {output_path}")
    print(
        f"Test_MSE-{mse:.4f}, Test_CI-{cindex:.4f}, "
        f"Test_RM2-{rm2:.4f}, Test_Pearson-{pearson:.4f}"
    )


if __name__ == "__main__":
    main()
