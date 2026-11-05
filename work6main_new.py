import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch.optim as optim
import torch
from metrics import get_cindex,get_rm2,get_pearson_np
from work6dataset_new import *
from work6MSNmodel_new import MSN_DTA
from utils import *
from log.train_logger import TrainLogger
import torch.nn as nn
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import math

##########################

Data_root = "../data",
save_dir = "save",
Dataset = 'davis-DualPGdq(Moe)',
save_model = '../code/output/model',
load_model_path = None
NUM_EPOCHS = 600
steps_per_epoch = 100

######################################

params = dict(
    data_root="../data",
    save_dir="save",
    dataset='davis-DualPGdq(Moe)',
    save_model='../code/output/model',
    lr=5e-4,
    batch_size=128
)

logger = TrainLogger(params)
logger.info(__file__)

def main():
    fpath = os.path.join("../data", 'davis-DualPGdq(Moe)')
    train_set = PreDataset(fpath, train=True)
    test_set = PreDataset(fpath, train=False)
    break_flag = False
    PreTrain = False
    early_stop = 80
    reset_epochs = 999

    print('The train data number is {}'.format(len(train_set)))
    print('The test data number is {}'.format(len(test_set)))

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device('cuda:0')
    model = MSN_DTA(3, 22, 25 + 1, embedding_size=128,
                    filter_num=32, out_dim=1, blpretrain=PreTrain).to(device)

    if load_model_path is not None:
        print('load_model...', load_model_path)
        load_model = torch.load(load_model_path)
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in load_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.MSELoss()

    run_MSE = AverageMeter()
    run_best_MSE = BestMeter('min')
    run_best_RM2 = BestMeter('max')

    model.train()

    for epoch in range(NUM_EPOCHS):
        if break_flag:
            break

        # ---------- Training ----------
        for data in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
            data = data.to(device)
            pred = model(data, epoch + 1, reset_epochs)
            mse_loss = criterion(pred.view(-1), data.y.view(-1))

            optimizer.zero_grad()
            mse_loss.backward()
            optimizer.step()

            run_MSE.update(mse_loss.item(), data.y.size(0))

        train_mse_epoch = run_MSE.get_average()
        run_MSE.reset()

        # ---------- Validation ----------
        if (epoch + 1) % reset_epochs == 0:
            print("Update node feature!")
            continue
        else:
            test_loss, test_cindex, test_rm2, test_pearson = val(model, criterion, test_loader, device)

            message = (f"Epoch-{epoch+1}, Train_MSE-{train_mse_epoch:.4f}, "
                       f"Test_MSE-{test_loss:.4f}, Test_CI-{test_cindex:.4f}, "
                       f"Test_RM2-{test_rm2:.4f}, Test_Pearson-{test_pearson:.4f}")
            logger.info(message)

            results_path = os.path.join(logger.get_model_dir(), "epoch_results.txt")
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(message + "\n")

            if test_loss < run_best_MSE.get_best():
                run_best_MSE.update(test_loss)
                if params.get('save_model', None):
                    save_model_dict(model, logger.get_model_dir(), message)
            else:
                count = run_best_MSE.counter()
                if count > early_stop:
                    logger.info(f"early stop in epoch {epoch + 1}")
                    break_flag = True

def val(model, criterion, dataloader, device):
    model.eval()
    running_loss = AverageMeter()
    y_trues = []
    y_preds = []

    for data in dataloader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data, 1, 9999)
            loss = criterion(pred.view(-1), data.y.view(-1))
            running_loss.update(loss.item(), data.y.size(0))

            y_true = data.y.detach().cpu().numpy().reshape(-1)
            y_pred = pred.detach().cpu().numpy().reshape(-1)

            y_trues.append(y_true)
            y_preds.append(y_pred)

    if len(y_trues) == 0:
        model.train()
        return None, None, None, None

    y_trues = np.concatenate(y_trues, axis=0)
    y_preds = np.concatenate(y_preds, axis=0)

    test_loss = running_loss.get_average()
    test_cindex = get_cindex(y_trues, y_preds)
    test_rm2 = get_rm2(y_trues, y_preds)
    test_pearson = get_pearson_np(y_trues, y_preds)

    model.train()
    return test_loss, test_cindex, test_rm2, test_pearson


if __name__ == "__main__":
    main()
#export LD_LIBRARY_PATH=/home/dell/anaconda3/envs/msn_dta/lib:$LD_LIBRARY_PATH
