#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from module.nets import VAE, Classifier, _Module, device
from module.nets2 import VAE as VAE2
from tqdm import tqdm

from loader import N_CLASS, C, shuffle_batching, tr, ts


def plt_df(df):
    df.set_index("time").plot()
    plt.show()

def get_rows_dup_time(df, col = "time"):
    t = [g for _, g in df.groupby(col) if len(g) > 1]
    if t:
        return pd.concat(t)
    print(f"no dup col: {col}")


# NOTE: foward pass test
# n_latent = 6; z_dim = 64
n_latent = 4; z_dim = 128
mod = VAE(i_c = C, n_latent = n_latent, z_dim = z_dim, name = "vae")
# mod = VAE2(i_c = C, n_latent = 4, name = "vae2")
m = mod.to(device)
print(f'Model "{mod.name}" -> {sum(p.numel() for p in mod.parameters()):,} params')

print(f'n_tr, n_ts: {len(tr), len(ts)}')
# d1, mu, var, loss = m(tr[0])    # XXX


# NOTE: train

# NOTE: VAE training

from torch import optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import ChainedScheduler, CosineAnnealingLR

N_EPOCH = 200 # 5
LR = 1E-3

LOG_EVERY  = 10
EVAL_EVERY = 10

def train_vae(model: _Module, scheduler):

    def vae_train(model, optimizer, tr):

        tot_loss = 0.0
        ys = []
        n = len(tr)

        model.train()
        for i, item in enumerate(shuffle_batching(tr)):
            # iputs, targs = item.iputs, item.iputs
            optimizer.zero_grad(set_to_none=True) # lower memory footprint
            preds, mu, var, loss = model(item)
            loss.backward()
            clip_grad_norm_(model.parameters(),
                            max_norm = 1.0,
                            norm_type = 2.0) # NOTE: L2 regularisation, clip norm
            optimizer.step()

            tot_loss += loss.item()
            ys += [preds]

        return tot_loss / n, ys

    def vae_eval(model, ts):

        tot_loss = 0.0
        ys = []
        n = len(ts)

        model.eval()
        for i, item in enumerate(shuffle_batching(ts)):
            # iputs, targs = item.iputs, item.iputs

            preds, mu, var, loss = model(item)
            tot_loss += loss.item()
            ys += [preds]

        return tot_loss / n, ys

    tr_losses, ts_losses = [], []

    for e in tqdm(range(N_EPOCH)):
        lr = scheduler.get_last_lr()[-1] # lr delta; returns a list

        loss_tr, tr_preds = vae_train(model, scheduler.optimizer, tr)
        tr_losses += [loss_tr]

        if e % LOG_EVERY == 0:
            print(f'-> tr epoch: [{e:>{len(str(N_EPOCH-1))}}/{N_EPOCH-1}],',
                f'loss: {loss_tr:.5f},',
                f'lr: {lr:.5f}', sep=' ', end='\t')
            print("\n", end='')

        if e % EVAL_EVERY == 0:
            loss_ts, ts_preds = vae_eval(model, ts)
            ts_losses += [loss_ts]
            print(f'-> ts epoch: [{e:>{len(str(N_EPOCH-1))}}/{N_EPOCH-1}],',
                f'loss: {loss_ts:.5f},', sep=' ', end='\t')
            print("\n", end='')

        scheduler.step()

    return tr_losses, ts_losses

opti_1 = optim.AdamW(mod.parameters(), lr = LR)
scheduler_1 = ChainedScheduler([
    # CosineAnnealingWarmRestarts(opti_1, T_0=EPOCHS//2, # T_0=n_epoch means no restart
    #                             T_mult=1, eta_min=LR/1E2),
    CosineAnnealingLR(opti_1, T_max = N_EPOCH * 2, eta_min = LR/1E2)])

tr_losses_vae, ts_losses_vae = train_vae(model = m, scheduler = scheduler_1)


# NOTE: classifier
import torch.nn.functional as F

N_EPOCH = 1000 # 300
LR = 1E-2

def train_classifier(model: _Module, optimizer):

    # categorical cross-entropy loss:
    loss_fn = lambda y, Y: F.nll_loss(F.log_softmax(y, dim = 1), Y)

    def classifier_train(model, optimizer, tr):

        tot_loss = 0.0
        acc = []
        n = len(tr)

        model.train()
        for _, item in enumerate(shuffle_batching(tr)):
            iputs, targs = item.iputs, item.targs

            optimizer.zero_grad()
            # preds, loss = model(item)

            preds = model(item)
            loss = loss_fn(preds, targs)
            loss.backward()
            optimizer.step()
            acc += [((F.softmax(preds, dim = 1).max(1)[1] == targs).sum() / len(targs)).item()]

            tot_loss += loss.item()

        return tot_loss / n, acc

    def classifier_eval(model, ts):

        tot_loss = 0.0
        acc = []
        n = len(tr)

        model.eval()
        for _, item in enumerate(shuffle_batching(ts)):
            # preds, loss = model(item)

            iputs, targs = item.iputs, item.targs
            preds = model(item)
            loss = loss_fn(preds, targs)
            acc += [((F.softmax(preds, dim = 1).max(1)[1] == targs).sum() / len(targs)).item()]

            tot_loss += loss.item()

        return tot_loss / n, acc

    tr_losses, ts_losses = [], []
    for e in tqdm(range(N_EPOCH)):

        tr_loss, tr_acc = classifier_train(model, optimizer, tr)
        tr_losses += [tr_loss]

        if e % LOG_EVERY == 0:
            print(f"-> tr: [{e}] loss: {tr_loss:.5f} "
                f"acc: {np.mean(tr_acc):.5f} ")

        if e % EVAL_EVERY == 0:
            ts_loss, ts_acc = classifier_eval(model, ts)
            ts_losses += [ts_loss]
            print(f"-> ts: [{e}] loss: {ts_loss:.5f}"
                f"acc: {np.mean(tr_acc):.5f} ")

    return tr_losses, ts_losses

classifier = Classifier(n_class = N_CLASS, vae = m, n_latent = n_latent, name = "classifier")
m_class = classifier.to(device)

optimizer = optim.SGD(m_class.parameters(), lr = LR)

tr_losses_class, ts_losses_class = train_classifier(m_class, optimizer)


# m.save_checkpoint()
# m_class.save_checkpoint()

# from module.util import pkl_dmp, pkl_lod

# pkl_dmp("oput/tr_losses_vae.pkl", pd.DataFrame(tr_losses_vae))
# pkl_dmp("oput/ts_losses_vae.pkl", pd.DataFrame(ts_losses_vae))
# pkl_dmp("oput/tr_losses_class.pkl", pd.DataFrame(tr_losses_class))
# pkl_dmp("oput/ts_losses_class.pkl", pd.DataFrame(ts_losses_class))

# pkl_dmp("oput/data_tr.pkl", tr)
# pkl_dmp("oput/data_ts.pkl", ts)


# def _smooth_out_values(data, window_size=10):
#     return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
