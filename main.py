import pickle
import os
import pandas as pd
import torch
from tqdm import tqdm
from src.models import *
from src.constants import *
from src.plotting import *
from src.pot import *
from src.utils import *
from src.diagnosis import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from time import time
from pprint import pprint
import psutil
import numpy as np
import matplotlib.colors as mcolors
# import time
from collections import defaultdict


def convert_to_windows(data, model):
    windows = []
    w_size = model.n_window
    for i, g in enumerate(data):
        if i >= w_size:
            w = data[i - w_size:i]  # cut
        else:
            w = torch.cat([data[0].repeat(w_size - i, 1), data[0:i]])  # pad
        if 'PUC' in model.name or 'DTAAD' in model.name or 'Attention' in model.name or 'TranAD' in model.name:
            windows.append(w)
        else:
            windows.append(w.view(-1))
    return torch.stack(windows)


def load_dataset(dataset):
    folder = os.path.join(output_folder, dataset)
    if not os.path.exists(folder):
        raise Exception('Processed Data not found.')
    loader = []
    for file in ['train', 'test', 'labels']:

        if dataset == 'SMD': file = 'machine-1-6_' + file
        if dataset == 'SMAP': file = 'P-1_' + file
        if dataset == 'MSL': file = 'C-1_' + file
        if dataset == 'UCR': file = '136_' + file
        if dataset == 'NAB': file = 'ec2_request_latency_system_failure_' + file
        loader.append(np.load(os.path.join(folder, f'{file}.npy')))
    if args.less: loader[0] = cut_array(0.2, loader[0])
    train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
    test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
    labels = loader[2]
    return train_loader, test_loader, labels


def save_model(model, optimizer, scheduler, epoch, accuracy_list):
    folder = f'checkpoints/{args.model}_{args.dataset}/'
    os.makedirs(folder, exist_ok=True)
    file_path = f'{folder}/model.ckpt'

    device = next(model.parameters()).device
    if device.type != 'cpu':
        model.to('cpu')

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list
    }, file_path)

    if device.type != 'cpu':
        model.to(device)


# def load_model(modelname, dims, dataset, save_path, feature_names):
#     import src.models
#     model_class = getattr(src.models, modelname)
#     model = model_class(dims, dataset, save_path, feature_names).double()
#     optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=1e-5)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
#     fname = f'checkpoints/{args.model}_{args.dataset}/model.ckpt'
#
#     if os.path.exists(fname) and (not args.retrain or args.test):
#         print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
#         checkpoint = torch.load(fname)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
#         epoch = checkpoint['epoch']
#         accuracy_list = checkpoint['accuracy_list']
#     else:
#         print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
#         epoch = -1
#         accuracy_list = []
#
#     if model.name == 'iTransformer':
#         model.to(torch.device(args.Device))
#
#     return model, optimizer, scheduler, epoch, accuracy_list


def load_model(modelname, dims, dataset, save_path, feature_names):
    import src.models
    model_class = getattr(src.models, modelname)

    # 根据模型名称传递不同参数
    if modelname == 'PUC':
        model = model_class(dims, dataset, save_path, feature_names).double()
    else:
        model = model_class(dims).double()  # 其他模型保持原有初始化方式

    optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    fname = f'checkpoints/{args.model}_{args.dataset}/model.ckpt'

    if os.path.exists(fname) and (not args.retrain or args.test):
        print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        accuracy_list = checkpoint['accuracy_list']
    else:
        print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
        epoch = -1
        accuracy_list = []

    if model.name == 'iTransformer':
        model.to(torch.device(args.Device))

    return model, optimizer, scheduler, epoch, accuracy_list


def get_system_stats():
    """获取系统资源使用情况"""
    stats = {}
    stats['cpu_percent'] = psutil.cpu_percent()
    stats['cpu_mem_percent'] = psutil.virtual_memory().percent

    if torch.cuda.is_available():
        stats['gpu_percent'] = torch.cuda.utilization(0)
        stats['gpu_mem_used'] = torch.cuda.memory_allocated(0) / (1024 ** 2)
        stats['gpu_mem_total'] = torch.cuda.memory_reserved(0) / (1024 ** 2)
    else:
        stats['gpu_percent'] = 0
        stats['gpu_mem_used'] = 0
        stats['gpu_mem_total'] = 0

    return stats


def backprop(epoch, model, data, dataO, optimizer, scheduler, training=True):
    l = nn.MSELoss(reduction='mean' if training else 'none')
    feats = dataO.shape[1]
    if 'DAGMM' in model.name:
        # optimizer = optimizer.to('cuda')
        l = nn.MSELoss(reduction='none')
        model.to(torch.device(args.Device))
        compute = ComputeLoss(model, 0.1, 0.005, 'cuda', model.n_gmm)
        n = epoch + 1
        w_size = model.n_window
        l1s = []
        l2s = []
        if training:
            for d in data:
                d = d.to(torch.device(args.Device))
                _, x_hat, z, gamma = model(d)
                l1, l2 = l(x_hat, d), l(gamma, d)
                l1s.append(torch.mean(l1).item())
                l2s.append(torch.mean(l2).item())
                loss = torch.mean(l1) + torch.mean(l2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
            return np.mean(l1s) + np.mean(l2s), optimizer.param_groups[0]['lr']
        else:
            model.to(torch.device('cpu'))
            ae1s = []
            for d in data:
                _, x_hat, _, _ = model(d)
                ae1s.append(x_hat)
            ae1s = torch.stack(ae1s)
            y_pred = ae1s[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            loss = l(ae1s, data)[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            return loss.detach().numpy(), y_pred.detach().numpy()
            # return loss.to("cpu").detach().numpy(), y_pred.to("cpu").detach().numpy()
    if 'Attention' in model.name:
        l = nn.MSELoss(reduction='none')
        model.to(torch.device(args.Device))
        n = epoch + 1
        w_size = model.n_window
        l1s = []
        res = []
        if training:
            for d in data:
                d = d.to(torch.device(args.Device))
                ae = model(d)
                # res.append(torch.mean(ats, axis=0).view(-1))
                l1 = l(ae, d)
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # res = torch.stack(res); np.save('ascores.npy', res.detach().numpy())
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            model.to(torch.device(args.Device))
            ae1s, y_pred = [], []
            for d in data:
                ae1 = model(d)
                y_pred.append(ae1[-1])
                ae1s.append(ae1)
            ae1s, y_pred = torch.stack(ae1s), torch.stack(y_pred)
            loss = torch.mean(l(ae1s, data), axis=1)
            return loss.detach().numpy(), y_pred.detach().numpy()
    elif 'OmniAnomaly' in model.name:
        if training:
            mses, klds = [], []
            model.to(torch.device('cpu'))
            for i, d in enumerate(data):
                d = d.to(torch.device('cpu'))
                y_pred, mu, logvar, hidden = model(d, hidden if i else None)
                MSE = l(y_pred, d)
                KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
                loss = MSE + model.beta * KLD
                mses.append(torch.mean(MSE).item())
                klds.append(model.beta * torch.mean(KLD).item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tKLD = {np.mean(klds)}')
            scheduler.step()
            return loss.item(), optimizer.param_groups[0]['lr']
        else:
            model.to(torch.device('cpu'))
            y_preds = []
            for i, d in enumerate(data):
                y_pred, _, _, hidden = model(d, hidden if i else None)
                y_preds.append(y_pred)
            y_pred = torch.stack(y_preds)
            MSE = l(y_pred, data)
            return MSE.detach().numpy(), y_pred.detach().numpy()
    elif 'USAD' in model.name:
        l = nn.MSELoss(reduction='none')
        model.to(torch.device(args.Device))
        n = epoch + 1
        w_size = model.n_window
        l1s, l2s = [], []
        if training:
            for d in data:
                d = d.to(torch.device(args.Device))
                ae1s, ae2s, ae2ae1s = model(d)
                l1 = (1 / n) * l(ae1s, d) + (1 - 1 / n) * l(ae2ae1s, d)
                l2 = (1 / n) * l(ae2s, d) - (1 - 1 / n) * l(ae2ae1s, d)
                l1s.append(torch.mean(l1).item());
                l2s.append(torch.mean(l2).item())
                loss = torch.mean(l1 + l2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
            return np.mean(l1s) + np.mean(l2s), optimizer.param_groups[0]['lr']
        else:
            model.to(torch.device('cpu'))
            ae1s, ae2s, ae2ae1s = [], [], []
            for d in data:
                ae1, ae2, ae2ae1 = model(d)
                ae1s.append(ae1)
                ae2s.append(ae2)
                ae2ae1s.append(ae2ae1)
            ae1s, ae2s, ae2ae1s = torch.stack(ae1s), torch.stack(ae2s), torch.stack(ae2ae1s)
            y_pred = ae1s[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            loss = 0.2 * l(ae1s, data) + 0.8 * l(ae2ae1s, data)
            loss = loss[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            return loss.detach().numpy(), y_pred.detach().numpy()
    elif model.name in ['GDN', 'MTAD_GAT', 'MSCRED', 'CAE_M']:
        l = nn.MSELoss(reduction='none')
        model.to(torch.device('cpu'))
        n = epoch + 1
        w_size = model.n_window
        l1s = []
        if training:
            for i, d in enumerate(data):
                d = d.to(torch.device('cpu'))
                if 'MTAD_GAT' in model.name:
                    x, h = model(d, h if i else None)
                else:
                    x = model(d)
                loss = torch.mean(l(x, d))
                l1s.append(torch.mean(loss).item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            model.to(torch.device('cpu'))
            xs = []
            for d in data:
                if 'MTAD_GAT' in model.name:
                    x, h = model(d, None)
                else:
                    x = model(d)
                xs.append(x)
            xs = torch.stack(xs)
            y_pred = xs[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            loss = l(xs, data)
            loss = loss[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            return loss.detach().numpy(), y_pred.detach().numpy()
            # return loss.to("cpu").detach().numpy(), y_pred.to("cpu").detach().numpy()
    elif 'GAN' in model.name:
        l = nn.MSELoss(reduction='none')
        model.to(torch.device('cpu'))
        bcel = nn.BCELoss(reduction='mean')
        msel = nn.MSELoss(reduction='mean')
        real_label, fake_label = torch.tensor([0.9]), torch.tensor([0.1])  # label smoothing
        real_label, fake_label = real_label.type(torch.DoubleTensor), fake_label.type(torch.DoubleTensor)
        n = epoch + 1;
        w_size = model.n_window
        mses, gls, dls = [], [], []
        if training:
            for d in data:
                # training discriminator
                d = d.to(torch.device('cpu'))
                model.discriminator.zero_grad()
                _, real, fake = model(d)
                dl = bcel(real, real_label) + bcel(fake, fake_label)
                dl.backward()
                model.generator.zero_grad()
                optimizer.step()
                # training generator
                z, _, fake = model(d)
                mse = msel(z, d)
                gl = bcel(fake, real_label)
                tl = gl + mse
                tl.backward()
                model.discriminator.zero_grad()
                optimizer.step()
                mses.append(mse.item())
                gls.append(gl.item())
                dls.append(dl.item())
            # tqdm.write(f'Epoch {epoch},\tMSE = {mse},\tG = {gl},\tD = {dl}')
            tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tG = {np.mean(gls)},\tD = {np.mean(dls)}')
            return np.mean(gls) + np.mean(dls), optimizer.param_groups[0]['lr']
        else:
            model.to(torch.device('cpu'))
            outputs = []
            for d in data:
                z, _, _ = model(d)
                outputs.append(z)
            outputs = torch.stack(outputs)
            y_pred = outputs[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            loss = l(outputs, data)
            loss = loss[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            return loss.detach().numpy(), y_pred.detach().numpy()
    elif 'TranAD' in model.name:
        l = nn.MSELoss(reduction='none')
        model.to(torch.device(args.Device))
        data_x = torch.DoubleTensor(data)
        dataset = TensorDataset(data_x, data_x)
        bs = model.batch if training else len(data)
        dataloader = DataLoader(dataset, batch_size=bs)
        n = epoch + 1
        w_size = model.n_window
        l1s, l2s = [], []
        if training:
            for d, _ in dataloader:
                d = d.to(torch.device(args.Device))
                local_bs = d.shape[0]
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, local_bs, feats)
                z = model(window, elem)
                l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1 / n) * l(z[1], elem)
                if isinstance(z, tuple): z = z[1]
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            model.to(torch.device(args.Device))
            for d, _ in dataloader:
                d = d.to(torch.device(args.Device))
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, bs, feats)
                z = model(window, elem)
                if isinstance(z, tuple): z = z[1]
            loss = l(z, elem)[0]
            return loss.to('cpu').detach().numpy(), z.to('cpu').detach().numpy()[0]
    elif 'DTAAD' in model.name:
        l = nn.MSELoss(reduction='none')
        _lambda = 0.8
        model.to(torch.device(args.Device))
        data_x = torch.DoubleTensor(data)
        dataset = TensorDataset(data_x, data_x)
        bs = model.batch if training else len(data)
        dataloader = DataLoader(dataset, batch_size=bs)
        n = epoch + 1
        w_size = model.n_window
        l1s, l2s = [], []
        if training:
            for d, _ in dataloader:
                d = d.to(torch.device(args.Device))
                local_bs = d.shape[0]
                window = d.permute(0, 2, 1)
                elem = window[:, :, -1].view(1, local_bs, feats)
                z = model(window)
                l1 = _lambda * l(z[0].permute(1, 0, 2), elem) + (1 - _lambda) * l(z[1].permute(1, 0, 2), elem)
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            model.to(torch.device('cpu'))
            # model.to(torch.device(args.Device))
            for d, _ in dataloader:
                d = d.to(torch.device(args.Device))
                window = d.permute(0, 2, 1)
                elem = window[:, :, -1].view(1, bs, feats)
                z = model(window)
                z = z[1].permute(1, 0, 2)
            loss = l(z, elem)[0]
            return loss.detach().numpy(), z.detach().numpy()[0]
    elif 'PUC' in model.name:
        l = nn.MSELoss(reduction='none')
        _lambda = 0.8
        # model.to(torch.device('cpu'))
        model.to(torch.device(args.Device))
        data_x = torch.DoubleTensor(data)
        dataset = TensorDataset(data_x, data_x)
        bs = model.batch if training else len(data)
        dataloader = DataLoader(dataset, batch_size=bs)
        n = epoch + 1
        w_size = model.n_window
        l1s, l2s = [], []
        if training:
            for d, _ in dataloader:
                # d = d.to(torch.device('cpu'))
                d = d.to(torch.device(args.Device))
                local_bs = d.shape[0]
                window = d.permute(0, 2, 1)
                elem = window[:, :, -1].view(1, local_bs, feats)
                # a = data[1, :, :]
                z = model(window)
                # l1 = (1 - _lambda) * l(z[0].permute(1, 0, 2), elem) + _lambda * l(z[1].permute(1, 0, 2), elem)
                l1 = l(z.permute(1, 0, 2), elem)
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            # model.to(torch.device('cpu'))
            model.to(torch.device(args.Device))
            for d, _ in dataloader:
                d = d.to(torch.device(args.Device))
                window = d.permute(0, 2, 1)
                elem = window[:, :, -1].view(1, bs, feats)
                z = model(window)
                # z = z[1].permute(1, 0, 2)
                z = z.permute(1, 0, 2)
            # loss = ((1 - _lambda) * l(z[0].permute(1, 0, 2), elem) + _lambda * l(z[1].permute(1, 0, 2), elem))[0]
            loss = l(z, elem)[0]
            # return loss.detach().numpy(), z.detach().numpy()[0]
            # return loss.to("cpu").detach().numpy(), z[1].permute(1, 0, 2).to("cpu").detach().numpy()[0]
            return loss.to("cpu").detach().numpy(), z.to("cpu").detach().numpy()[0]
    else:

        model.to(torch.device('cpu'))
        # data = data.to(torch.device(args.Device))
        y_pred = model(data)
        loss = l(y_pred, data)
        if training:
            tqdm.write(f'Epoch {epoch},\tMSE = {loss}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            return loss.item(), optimizer.param_groups[0]['lr']
        else:
            model.to(torch.device('cpu'))
            return loss.detach().numpy(), y_pred.detach().numpy()


# def set_seed(seed=2):
#     """设置全局随机种子以确保结果可复现"""
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


def plot_anomaly_radar(feature_names, anomaly_intensities, title, save_path):
    """
    绘制异常模式雷达图
    :param feature_names: 特征名称列表（对应雷达图的轴标签）
    :param anomaly_intensities: 各特征维度的异常强度（需与 feature_names 长度一致）
    :param title: 图表标题（如 "Anomaly Pattern - machine2-6"）
    :param save_path: 图片保存路径
    """
    num_dims = len(feature_names)
    # 计算雷达图角度，让第一个维度从正右方开始，均匀分布
    angles = np.linspace(0, 2 * np.pi, num_dims, endpoint=False).tolist()
    # 闭合图形（让最后一个点与第一个点相连）
    anomaly_intensities = np.concatenate((anomaly_intensities, [anomaly_intensities[0]]))
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)

    # 绘制雷达图轮廓
    ax.fill(angles, anomaly_intensities, color=mcolors.TABLEAU_COLORS['tab:blue'], alpha=0.25, label='Anomaly Pattern')
    ax.plot(angles, anomaly_intensities, color=mcolors.TABLEAU_COLORS['tab:blue'], linewidth=2)

    # 设置轴标签（特征名称）
    ax.set_thetagrids(np.degrees(angles[:-1]), feature_names, fontsize=10)
    # 设置雷达图范围（根据数据调整，让图形更美观）
    ax.set_ylim(0, np.max(anomaly_intensities) * 1.1)
    ax.set_title(title, y=1.05, fontsize=16)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':

    # 在数据加载前设置随机种子
    # set_seed(2)  # 使用42作为示例种子，可替换为其他值

    train_loader, test_loader, labels = load_dataset(args.dataset)

    # 获取特征名称（从数据中推断或手动指定）
    feature_names = [f"feature_{i}" for i in range(labels.shape[1])]

    # 创建保存路径
    save_path = f"./results/{args.dataset}/"
    os.makedirs(save_path, exist_ok=True)

    # 加载模型，传递数据集名称、保存路径和特征名称
    model, optimizer, scheduler, epoch, accuracy_list = load_model(
        args.model, labels.shape[1], args.dataset, save_path, feature_names
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params}')

    ## Prepare data
    trainD, testD = next(iter(train_loader)), next(iter(test_loader))
    trainO, testO = trainD, testD
    if model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT',
                      'MAD_GAN', 'TranAD', 'DTAAD'] or 'PUC' in model.name:
        trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)
        train1 = convert_to_windows(trainO, model)

    ### Training phase
    # if not args.test:
    #     print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
    #     num_epochs = 5
    #     e = epoch + 1
    #     start_time = time.time()
    #     for e in tqdm(list(range(e, num_epochs + e + 1))):
    #         epoch_start = time.time()
    #         lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler)
    #         accuracy_list.append((lossT, lr))
    #         print(
    #             color.BOLD + 'Epoch training time: ' + "{:10.4f}".format(time.time() - epoch_start) + ' s' + color.ENDC)
    #     print(color.BOLD + 'Total training time: ' + "{:10.4f}".format(time.time() - start_time) + ' s' + color.ENDC)
    #     save_model(model, optimizer, scheduler, e, accuracy_list)
    #     plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')
    #
    # ### Testing phase with performance monitoring
    # torch.zero_grad = True
    # model.eval()
    # print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
    #
    # # Initialize performance tracking
    # perf_stats = defaultdict(list)
    # device = next(model.parameters()).device
    # test_results = []
    #
    # # 设置采样频率，每100个样本收集一次系统信息
    # sampling_freq = 300
    #
    # # Warm-up
    # with torch.no_grad():
    #     _ = backprop(0, model, testD[:1], testO[:1], optimizer, scheduler, training=False)
    #
    # # Testing loop - 优化版本
    # test_start = time.time()
    # with torch.no_grad():
    #     # 预先获取初始系统状态
    #     initial_stats = get_system_stats()
    #     prev_stats = initial_stats
    #
    #     # 计算批次数
    #     batch_size = 128  # 根据你的GPU内存调整
    #     num_batches = len(testD) // batch_size + (1 if len(testD) % batch_size != 0 else 0)
    #
    #     for batch_idx in tqdm(range(num_batches)):
    #         start_idx = batch_idx * batch_size
    #         end_idx = min((batch_idx + 1) * batch_size, len(testD))
    #
    #         # 获取当前批次的样本
    #         batch_testD = testD[start_idx:end_idx]
    #         batch_testO = testO[start_idx:end_idx]
    #
    #         # 批次处理时间
    #         batch_start = time.time()
    #         batch_losses = []
    #         batch_preds = []
    #
    #         # 批处理样本
    #         for i in range(len(batch_testD)):
    #             loss, y_pred = backprop(0, model, batch_testD[i:i + 1], batch_testO[i:i + 1], optimizer, scheduler,
    #                                     training=False)
    #             batch_losses.append(loss)
    #             batch_preds.append(y_pred)
    #
    #         batch_time = time.time() - batch_start
    #
    #         # 每sampling_freq个批次收集一次系统信息
    #         if batch_idx % sampling_freq == 0:
    #             current_stats = get_system_stats()
    #
    #             # 记录性能指标 - 使用批次平均值
    #             perf_stats['batch_idx'].append(batch_idx)
    #             perf_stats['time'].append(batch_time / len(batch_testD))  # 每个样本的平均时间
    #             perf_stats['cpu_usage'].append((prev_stats['cpu_percent'] + current_stats['cpu_percent']) / 2)
    #             perf_stats['memory_usage'].append(
    #                 (prev_stats['cpu_mem_percent'] + current_stats['cpu_mem_percent']) / 2)
    #
    #             if torch.cuda.is_available():
    #                 perf_stats['gpu_usage'].append((prev_stats['gpu_percent'] + current_stats['gpu_percent']) / 2)
    #                 perf_stats['gpu_memory'].append((prev_stats['gpu_mem_used'] + current_stats['gpu_mem_used']) / 2)
    #
    #             prev_stats = current_stats
    #
    #         # 收集结果
    #         test_results.extend(zip(batch_losses, batch_preds))
    #
    # # 最终系统状态
    # final_stats = get_system_stats()
    #
    # # 计算总体性能指标
    # total_samples = len(testD)
    # total_test_time = time.time() - test_start
    #
    # # 如果没有收集到系统统计信息，使用初始和最终状态
    # if not perf_stats['time']:
    #     perf_stats['time'].append(total_test_time / total_samples)
    #     perf_stats['cpu_usage'].append((initial_stats['cpu_percent'] + final_stats['cpu_percent']) / 2)
    #     perf_stats['memory_usage'].append((initial_stats['cpu_mem_percent'] + final_stats['cpu_mem_percent']) / 2)
    #
    #     if torch.cuda.is_available():
    #         perf_stats['gpu_usage'].append((initial_stats['gpu_percent'] + final_stats['gpu_percent']) / 2)
    #         perf_stats['gpu_memory'].append((initial_stats['gpu_mem_used'] + final_stats['gpu_mem_used']) / 2)
    #
    # # Combine results
    # all_loss = np.concatenate([r[0] for r in test_results])
    # all_y_pred = np.concatenate([r[1] for r in test_results])
    #
    # # Print performance summary
    # print("\n=== Performance Summary ===")
    # print(f"Total test samples: {total_samples}")
    # print(f"Total test time: {total_test_time:.2f} seconds")
    # print(f"Average time per sample: {np.mean(perf_stats['time']):.4f} ± {np.std(perf_stats['time']):.4f} seconds")
    # print(f"Max sample time: {max(perf_stats['time']):.4f} seconds")
    # print(f"Min sample time: {min(perf_stats['time']):.4f} seconds")
    #
    # print("\nCPU Usage:")
    # print(f"Average CPU usage: {np.mean(perf_stats['cpu_usage']):.1f}%")
    # print(f"Average memory usage: {np.mean(perf_stats['memory_usage']):.1f}%")
    #
    # if torch.cuda.is_available():
    #     print("\nGPU Usage:")
    #     print(f"Average GPU usage: {np.mean(perf_stats['gpu_usage']):.1f}%")
    #     print(f"Average GPU memory used: {np.mean(perf_stats['gpu_memory']):.1f} MB")
    #
    # # Save detailed performance data
    # perf_df = pd.DataFrame(perf_stats)
    # os.makedirs('results', exist_ok=True)
    # perf_csv_path = f'results/{args.model}_{args.dataset}_performance.csv'
    # perf_df.to_csv(perf_csv_path, index=False)
    # print(f"\nDetailed performance data saved to: {perf_csv_path}")
    #
    # ### Plot curves
    # if not args.test:
    #     if 'TranAD' in model.name or 'DTAAD' in model.name or 'PUC' in model.name:
    #         testO = torch.roll(testO, 1, 0)
    #     plotter(f'{args.model}_{args.dataset}', testO, all_y_pred, all_loss, labels)
    #
    # ## Plot attention
    # if not args.test and 'PUC' in model.name:
    #     plot_attention(model, 1, f'{args.model}_{args.dataset}')
    #
    # ### Scores
    # df = pd.DataFrame()
    # preds = []
    # lossT, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)
    # for i in range(all_loss.shape[1]):
    #     lt, l, ls = lossT[:, i], all_loss[:, i], labels[:, i]
    #     result, pred = pot_eval(lt, l, ls)
    #     preds.append(pred)
    #     result = pd.DataFrame(result, index=[0])
    #     df = pd.concat([df, result], ignore_index=True)
    #
    # lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(all_loss, axis=1)
    # labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
    # result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
    # result.update(hit_att(all_loss, labels))
    # result.update(ndcg(all_loss, labels))
    # result.update(CR(all_loss, labels))
    #
    # print("\n=== Evaluation Results ===")
    # print(df)
    # pprint(result)
    # ///////////////////////////////////////////////////////////////////////////////////////
    ### Training phase
    if not args.test:
        print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
        num_epochs = 5
        e = epoch + 1
        start = time()
        for e in tqdm(list(range(epoch + 1, e + num_epochs + 1))):
            lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler)
            accuracy_list.append((lossT, lr))
        print(color.BOLD + 'Training time: ' + "{:10.4f}".format(time() - start) + ' s' + color.ENDC)
        save_model(model, optimizer, scheduler, e, accuracy_list)
        plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')

    ### Testing phase
    torch.zero_grad = True
    model.eval()
    print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
    loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False)

    # ======================== 新增：计算维度异常强度，绘制雷达图 ========================
    if not args.test:
        # 假设 all_loss 是形状为 [时间步, 特征维度] 的异常分数矩阵
        # 按特征维度统计异常强度（示例：取均值，也可用最大值、总和等）
        anomaly_intensities = np.mean(loss, axis=0)

        # 特征名称（从 feature_names 获取）
        radar_feature_names = feature_names

        # 雷达图标题与保存路径
        radar_title = f"Anomaly Pattern - {args.dataset}"
        radar_save_path = f"{save_path}{args.dataset}_anomaly_radar.png"

        # 调用绘图函数
        plot_anomaly_radar(radar_feature_names, anomaly_intensities, radar_title, radar_save_path)
        print(f"异常模式雷达图已保存至: {radar_save_path}")

    ### Plot anomaly heatmap
    if not args.test:
        # 确保loss是numpy数组且在CPU上
        if isinstance(loss, torch.Tensor):
            loss = loss.cpu().numpy()

        # 生成时间戳
        timestamps = np.arange(loss.shape[0])

        # 调用绘图函数
        model.plot_dimension_heatmap(loss, timestamps)
        print(f"Anomaly heatmap saved to: {save_path}{args.dataset}_dimension_heatmap.png")

    ### Plot curves
    if not args.test:
        if 'TranAD' or 'DTAAD' or 'PUC' in model.name: testO = torch.roll(testO, 1, 0)
        plotter(f'{args.model}_{args.dataset}', testO, y_pred, loss, labels)

    ## Plot attention
    if not args.test:
        if 'DTAAD' or 'PUC' in model.name:
            plot_attention(model, 1, f'{args.model}_{args.dataset}')

    ### Scores
    df = pd.DataFrame()
    lossT, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)
    for i in range(0):
        lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
        result, pred = pot_eval(lt, l, ls)
        preds.append(pred)
        # df = df.concat(result, ignore_index=True)
        result = pd.DataFrame(result, index=[0])
        df = pd.concat([df, result], ignore_index=True)
    # preds = np.concatenate([i.reshape(-1, 1) + 0 for i in preds], axis=1)
    # pd.DataFrame(preds, columns=[str(i) for i in range(10)]).to_csv('labels.csv')
    lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
    labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
    result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
    result.update(hit_att(loss, labels))
    result.update(ndcg(loss, labels))
    print(df)
    pprint(result)
    # pprint(getresults2(df, result))


