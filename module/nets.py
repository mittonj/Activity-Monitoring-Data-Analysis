from os import environ
from os.path import isfile
from typing import NamedTuple, Sequence, Tuple, TypeAlias

import torch as tr
import torch.nn.functional as F
from torch import Tensor, nn

from .util import ensure_dir

device = tr.device("cuda" if tr.cuda.is_available() else "cpu")
if bool(environ.get("DEBUG", 0)):
    device = tr.device("cpu")

class Batch(NamedTuple):
    iputs: Tensor  # [N, T]
    targs: Tensor  # [N,]


Batches: TypeAlias = Sequence[Batch]


class _Module(nn.Module):

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    @tr.no_grad()
    def save_checkpoint(self):
        torchstate_fname = ensure_dir(f'model/{self.name}.trs')
        tr.save(self.state_dict(), torchstate_fname)
        print(f'saved model to: "{torchstate_fname}"')

    @tr.no_grad()
    def load_checkpoint(self) -> bool:
        torchstate_fname = f'model/{self.name}.trs'
        if isfile(torchstate_fname):
            print(f'loaded torch state: "{torchstate_fname}".')
            self.load_state_dict(tr.load(torchstate_fname))
            return True
        return False


class VAE(_Module):

    def __init__(self, i_c: int, n_latent: int,
                 z_dim: int,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        # conv
        self.encoder = nn.Sequential(
            nn.InstanceNorm1d(i_c, device = device),
            # 32, 16, 16
            nn.Conv1d(i_c, 32, kernel_size = 4, stride = 2, padding = 1,
                      device = device),
            nn.InstanceNorm1d(32, device = device),
            nn.LeakyReLU(),
            # 32, 8, 8
            nn.Conv1d(32, 64, kernel_size = 4, stride = 2, padding = 1,
                      device = device),
            nn.InstanceNorm1d(64, device = device),
            nn.LeakyReLU(),
        ) # -> (4, 64, 64)

        self.z_dim = z_dim
        # latent mean:
        self.z_mean = nn.Linear(in_features = 64 * self.z_dim,
                                out_features = n_latent,
                                device = device)
        # latent variance:
        self.z_var = nn.Linear(in_features = 64 * self.z_dim,
                               out_features = n_latent,
                               device = device)

        self.decode_z = nn.Linear(in_features = n_latent,
                                  out_features = 64 * self.z_dim,
                                  device = device)
        # transposed conv
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size = 3, stride = 2, padding = 0,
                               device = device),
            nn.InstanceNorm1d(32, device = device),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size = 3, stride = 2, padding = 1,
                               device = device),
            nn.InstanceNorm1d(16, device = device),
            nn.LeakyReLU(),
            nn.Conv1d(16, out_channels = i_c, kernel_size = 4, padding = 1,
                      device = device),
            nn.Tanh()
        )

    def encode(self, iput: Tensor) -> Tuple[Tensor, Tensor]:
        """ NCHW """
        iput = self.encoder(iput)
        iput = iput.view(iput.shape[0], -1)
        return self.z_mean(iput), self.z_var(iput)

    def _sample_z(self, mu: Tensor, log_variance: Tensor) -> Tensor:
        """aka. re-parameterization: we sample from mean/std of prob
        distributions instead of actual values.

        """
        std = tr.exp(.5 * log_variance)
        noise = tr.randn_like(std)
        return std * noise + mu

    def decode(self, z: Tensor) -> Tensor:
        oput = self.decode_z(z).view(z.shape[0], 64, self.z_dim)
        return self.decoder(oput)

    def forward(self, iput: Batch) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        mu, log_variance = self.encode(iput.iputs)
        z = self._sample_z(mu, log_variance)
        preds = self.decode(z)
        loss = vae_loss(preds, iput.iputs, mu, log_variance)
        return preds, mu, log_variance, loss


def z_loss(mu: Tensor, log_variance: Tensor) -> Tensor:
    """Latent loss.

    [r] section 3, https://arxiv.org/abs/1312.6114v10

    """
    return -(0.5 * (1. + log_variance - mu**2 - log_variance.exp()).sum(1)).mean(0)


def vae_loss(preds, iputs, mu, log_variance):
    # categorical cross-entropy loss:
    # loss_fn = lambda y, Y: F.nll_loss(F.log_softmax(y), Y)

    # Mean square error:
    loss_fn = lambda y, Y:  F.mse_loss(y, Y)
    loss_decomp = loss_fn(preds, iputs)
    loss_latent = z_loss(mu, log_variance)
    return loss_decomp + loss_latent


class Classifier(_Module):

    def __init__(self, n_class: int, vae: VAE, n_latent: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.vae = vae
        for p in vae.encoder.parameters():
            p.requires_grad = False

        self.classifier = nn.Linear(in_features = n_latent,
                                    out_features = n_class,
                                    device = device)

        # self.classifier = Conv(i_c = n_latent, o_c = n_class,
        #                        name = "conv")

        # self.classifier = BaseNet(i_c = n_latent, o_c = n_class,
        #                           name = "conv_v2")

    def forward(self, iput: Batch) -> Tensor:
        mu, var = self.vae.encode(iput.iputs)
        z = self.vae._sample_z(mu, var)
        # return self.classifier(z, iput.targs)
        return self.classifier(z)
