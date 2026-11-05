import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch.optim as optim
import torch
import pickle
from metrics import get_cindex,get_rm2,get_pearson_np
from work16dataset import *
#from absolute2 import BioSP_DTA
from work16model12 import BioSP_DTA
from dataloader import convert_batch_pr, drop_tokens
from utils import *
from log.train_logger import TrainLogger
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import math

##########################

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', 'data'))
DATASET = 'dualkiba-DualPGdq(work2)'

Data_root = DATA_ROOT
save_dir = "save",
Dataset = DATASET
save_model = os.path.join(BASE_DIR, 'output', 'model')
load_model_path = None
NUM_EPOCHS = 600
steps_per_epoch = 100

######################################

params = dict(
    data_root=DATA_ROOT,
    save_dir="save",
    dataset=DATASET,
    save_model=save_model,
    lr=5e-4,
    batch_size=256
)

logger = TrainLogger(params)
logger.info(__file__)

def collate_fn(data_list):
#按当前batch最大长度填充蛋白质序列特征，再按PyG方式合并图数据
    batch_pr = convert_batch_pr([data.target for data in data_list])
    data = Batch.from_data_list(data_list)
    data.target = batch_pr['input_ids']
    data.target_mask = batch_pr['attention_mask']
    return data

def main():
    fpath = os.path.join(DATA_ROOT, DATASET)
    train_set = PreDataset(fpath, train=True)
    test_set = PreDataset(fpath, train=False)
    break_flag = False
    PreTrain = False
    early_stop = 80
    reset_epochs = 999

    print('The train data number is {}'.format(len(train_set)))
    print('The test data number is {}'.format(len(test_set)))

    train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=4, pin_memory=True,
                              collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4, pin_memory=True,
                             collate_fn=collate_fn)

    c_path = os.path.join(fpath, 'processed', 'C_1280_2000_8.pkl') #路径文件
    C = pickle.load(open(c_path, 'rb')) #加载文件内容
    c = torch.from_numpy(C['cluster_centers']).permute(1, 0).to(dtype=torch.float32) #获取聚类中心
    text_c = C['text_cluster_centers'].to(dtype=torch.float32)
    aa_dict = C['aa'].to(dtype=torch.float32) #21个平均特征

    device = torch.device('cuda:0')
    aa_dict = aa_dict.to(device)
    #model = BioSP_DTA(3, 22, 25 + 1, embedding_size=128,filter_num=32, out_dim=1, c=c, blpretrain=PreTrain).to(device) #文本辅助聚类，需要传入文本中心，即text_c
    model = BioSP_DTA(3, 22, 25 + 1, embedding_size=128,filter_num=32, out_dim=1,c=c, text_c=text_c,blpretrain=PreTrain).to(device)

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
            batch_pr = {'input_ids': data.target, 'attention_mask': data.target_mask}
            data.target, _ = drop_tokens(batch_pr, 0.50, aa_avg_f=aa_dict, mutation_rate=0.20)
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

            if test_loss <= run_best_MSE.get_best():
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
#export LD_LIBRARY_PATH=/home/dell/anaconda3/envs/biosp_dta/lib:$LD_LIBRARY_PATH

