import math

import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import dgl.nn
# from dgl.mock_sparse import spspmm
from dgl.nn.pytorch import GATConv
from matplotlib import pyplot as plt

from torch.nn import TransformerEncoder, TransformerDecoder
from src.gltcn import *
from src.dlutils import *
from src.constants import *

import seaborn as sns

torch.manual_seed(1)
torch.cuda.manual_seed(1)


## Separate LSTM for each variable
class LSTM_Univariate(nn.Module):
    def __init__(self, feats):
        super(LSTM_Univariate, self).__init__()
        self.name = 'LSTM_Univariate'
        self.lr = 0.002
        self.n_feats = feats
        self.n_hidden = 1
        self.lstm = nn.ModuleList([nn.LSTM(1, self.n_hidden) for i in range(feats)])

    def forward(self, x):
        hidden = [(torch.rand(1, 1, self.n_hidden, dtype=torch.float64),
                   torch.randn(1, 1, self.n_hidden, dtype=torch.float64)) for i in range(self.n_feats)]
        outputs = []
        for i, g in enumerate(x):
            multivariate_output = []
            for j in range(self.n_feats):
                univariate_input = g.view(-1)[j].view(1, 1, -1)
                out, hidden[j] = self.lstm[j](univariate_input, hidden[j])
                multivariate_output.append(2 * out.view(-1))
            output = torch.cat(multivariate_output)
            outputs.append(output)
        return torch.stack(outputs)


## Simple Multi-Head Self-Attention Model
class Attention(nn.Module):
    def __init__(self, feats):
        super(Attention, self).__init__()
        self.name = 'Attention'
        self.lr = 0.0001
        self.n_feats = feats
        self.n_window = 5  # MHA w_size = 5
        self.n = self.n_feats * self.n_window
        self.atts = [nn.Sequential(nn.Linear(self.n, feats * feats),
                                   nn.ReLU(True)) for i in range(1)]
        self.atts = nn.ModuleList(self.atts)

    def forward(self, g):
        for at in self.atts:
            ats = at(g.view(-1)).reshape(self.n_feats, self.n_feats)
            g = torch.matmul(g, ats)
        return g, ats


## LSTM_NDT Model
class LSTM_NDT(nn.Module):
    def __init__(self, feats):
        super(LSTM_NDT, self).__init__()
        self.name = 'LSTM_NDT'
        self.lr = 0.002
        self.n_feats = feats
        self.n_hidden = 64
        self.lstm = nn.LSTM(feats, self.n_hidden)
        self.lstm2 = nn.LSTM(feats, self.n_feats)
        self.fcn = nn.Sequential(nn.Linear(self.n_feats, self.n_feats), nn.Sigmoid())

    def forward(self, x):
        hidden = (
            torch.rand(1, 1, self.n_hidden, dtype=torch.float64), torch.randn(1, 1, self.n_hidden, dtype=torch.float64))
        hidden2 = (
            torch.rand(1, 1, self.n_feats, dtype=torch.float64), torch.randn(1, 1, self.n_feats, dtype=torch.float64))
        outputs = []
        for i, g in enumerate(x):
            out, hidden = self.lstm(g.view(1, 1, -1), hidden)
            out, hidden2 = self.lstm2(g.view(1, 1, -1), hidden2)
            out = self.fcn(out.view(-1))
            outputs.append(2 * out.view(-1))
        return torch.stack(outputs)


## DAGMM Model (ICLR 18)
class DAGMM(nn.Module):
    def __init__(self, feats):
        super(DAGMM, self).__init__()
        self.name = 'DAGMM'
        self.lr = 0.0001
        self.beta = 0.01
        self.n_feats = feats
        self.n_hidden = 16
        self.n_latent = 8
        self.n_window = 10  # DAGMM w_size = 5
        self.n = self.n_feats * self.n_window
        self.n_gmm = self.n_feats * self.n_window
        self.encoder = nn.Sequential(
            nn.Linear(self.n, self.n_hidden), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n_latent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
        )
        self.estimate = nn.Sequential(
            nn.Linear(self.n_latent + 2, self.n_hidden), nn.Tanh(), nn.Dropout(0.1),
            nn.Linear(self.n_hidden, self.n_gmm), nn.Softmax(dim=1),
        )

    def compute_reconstruction(self, x, x_hat):
        relative_euclidean_distance = (x - x_hat).norm(2, dim=1) / x.norm(2, dim=1)
        cosine_similarity = F.cosine_similarity(x, x_hat, dim=1)
        return relative_euclidean_distance, cosine_similarity

    def forward(self, x):
        ## Encode Decoder
        x = x.view(1, -1)
        z_c = self.encoder(x)
        x_hat = self.decoder(z_c)
        ## Compute Reconstructoin
        rec_1, rec_2 = self.compute_reconstruction(x, x_hat)
        z = torch.cat([z_c, rec_1.unsqueeze(-1), rec_2.unsqueeze(-1)], dim=1)
        ## Estimate
        gamma = self.estimate(z)
        return z_c, x_hat.view(-1), z, gamma.view(-1)


## OmniAnomaly Model (KDD 19)
class OmniAnomaly(nn.Module):
    def __init__(self, feats):
        super(OmniAnomaly, self).__init__()
        self.name = 'OmniAnomaly'
        self.lr = 0.002
        self.beta = 0.01
        self.n_feats = feats
        self.n_hidden = 32
        self.n_latent = 8
        self.lstm = nn.GRU(feats, self.n_hidden, 2)
        self.encoder = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Flatten(),
            nn.Linear(self.n_hidden, 2 * self.n_latent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_feats), nn.Sigmoid(),
        )

    def forward(self, x, hidden=None):
        hidden = torch.rand(2, 1, self.n_hidden, dtype=torch.float64) if hidden is not None else hidden
        out, hidden = self.lstm(x.view(1, 1, -1), hidden)
        ## Encode
        x = self.encoder(out)
        mu, logvar = torch.split(x, [self.n_latent, self.n_latent], dim=-1)
        ## Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        x = mu + eps * std
        ## Decoder
        x = self.decoder(x)
        return x.view(-1), mu.view(-1), logvar.view(-1), hidden


## USAD Model (KDD 20)
class USAD(nn.Module):
    def __init__(self, feats):
        super(USAD, self).__init__()
        self.name = 'USAD'
        self.lr = 0.0001
        self.n_feats = feats
        self.n_hidden = 100
        self.n_latent = 5
        self.n_window = 10  # USAD w_size = 5
        self.n = self.n_feats * self.n_window
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_latent), nn.ReLU(True),
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
        )

    def forward(self, g):
        ## Encode
        z = self.encoder(g.view(1, -1))
        ## Decoders (Phase 1)
        ae1 = self.decoder1(z)
        ae2 = self.decoder2(z)
        ## Encode-Decode (Phase 2)
        ae2ae1 = self.decoder2(self.encoder(ae1))
        return ae1.view(-1), ae2.view(-1), ae2ae1.view(-1)


## MSCRED Model (AAAI 19)
class MSCRED(nn.Module):
    def __init__(self, feats):
        super(MSCRED, self).__init__()
        self.name = 'MSCRED'
        self.lr = 0.0001
        self.n_feats = feats
        self.n_window = feats
        self.encoder = nn.ModuleList([
            ConvLSTM(1, 32, (3, 3), 1, True, True, False),
            ConvLSTM(32, 64, (3, 3), 1, True, True, False),
            ConvLSTM(64, 128, (3, 3), 1, True, True, False),
        ]
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, (3, 3), 1, 1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, (3, 3), 1, 1), nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, (3, 3), 1, 1), nn.Sigmoid(),
        )

    def forward(self, g):
        ## Encode
        z = g.view(1, 1, self.n_feats, self.n_window)
        for cell in self.encoder:
            _, z = cell(z.view(1, *z.shape))
            z = z[0][0]
        ## Decode
        x = self.decoder(z)
        return x.view(-1)


## CAE-M Model (TKDE 21)
class CAE_M(nn.Module):
    def __init__(self, feats):
        super(CAE_M, self).__init__()
        self.name = 'CAE_M'
        self.lr = 0.001
        self.n_feats = feats
        self.n_window = feats
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, (3, 3), 1, 1), nn.Sigmoid(),
            nn.Conv2d(8, 16, (3, 3), 1, 1), nn.Sigmoid(),
            nn.Conv2d(16, 32, (3, 3), 1, 1), nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 4, (3, 3), 1, 1), nn.Sigmoid(),
            nn.ConvTranspose2d(4, 4, (3, 3), 1, 1), nn.Sigmoid(),
            nn.ConvTranspose2d(4, 1, (3, 3), 1, 1), nn.Sigmoid(),
        )

    def forward(self, g):
        ## Encode
        z = g.view(1, 1, self.n_feats, self.n_window)
        z = self.encoder(z)
        ## Decode
        x = self.decoder(z)
        return x.view(-1)


## MTAD_GAT Model (ICDM 20)
class MTAD_GAT(nn.Module):
    def __init__(self, feats):
        super(MTAD_GAT, self).__init__()
        self.name = 'MTAD_GAT'
        self.lr = 0.0001
        self.n_feats = feats
        self.n_window = feats
        self.n_hidden = feats * feats
        self.g = dgl.graph((torch.tensor(list(range(1, feats + 1))), torch.tensor([0] * feats)))
        self.g = dgl.add_self_loop(self.g)
        self.feature_gat = GATConv(feats, 1, feats)
        self.time_gat = GATConv(feats, 1, feats)
        self.gru = nn.GRU((feats + 1) * feats * 3, feats * feats, 1)

    def forward(self, data, hidden):
        hidden = torch.rand(1, 1, self.n_hidden, dtype=torch.float64) if hidden is not None else hidden
        data = data.view(self.n_window, self.n_feats)
        data_r = torch.cat((torch.zeros(1, self.n_feats), data))
        feat_r = self.feature_gat(self.g, data_r)
        data_t = torch.cat((torch.zeros(1, self.n_feats), data.t()))
        time_r = self.time_gat(self.g, data_t)
        data = torch.cat((torch.zeros(1, self.n_feats), data))
        data = data.view(self.n_window + 1, self.n_feats, 1)
        x = torch.cat((data, feat_r, time_r), dim=2).view(1, 1, -1)
        x, h = self.gru(x, hidden)
        return x.view(-1), h


## GDN Model (AAAI 21)
class GDN(nn.Module):
    def __init__(self, feats):
        super(GDN, self).__init__()
        self.name = 'GDN'
        self.lr = 0.0001
        self.n_feats = feats
        self.n_window = 5
        self.n_hidden = 16
        self.n = self.n_window * self.n_feats
        src_ids = np.repeat(np.array(list(range(feats))), feats)
        dst_ids = np.array(list(range(feats)) * feats)
        self.g = dgl.graph((torch.tensor(src_ids), torch.tensor(dst_ids)))
        self.g = dgl.add_self_loop(self.g)
        self.feature_gat = GATConv(1, 1, feats)
        self.attention = nn.Sequential(
            nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_window), nn.Softmax(dim=0),
        )
        self.fcn = nn.Sequential(
            nn.Linear(self.n_feats, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_window), nn.Sigmoid(),
        )

    def forward(self, data):
        # Bahdanau style attention
        att_score = self.attention(data).view(self.n_window, 1)
        data = data.view(self.n_window, self.n_feats)
        data_r = torch.matmul(data.permute(1, 0), att_score)
        # GAT convolution on complete graph
        feat_r = self.feature_gat(self.g, data_r)
        feat_r = feat_r.view(self.n_feats, self.n_feats)
        # Pass through a FCN
        x = self.fcn(feat_r)
        return x.view(-1)


# MAD_GAN (ICANN 19)
class MAD_GAN(nn.Module):
    def __init__(self, feats):
        super(MAD_GAN, self).__init__()
        self.name = 'MAD_GAN'
        self.lr = 0.0001
        self.n_feats = feats
        self.n_hidden = 16
        self.n_window = 5  # MAD_GAN w_size = 5
        self.n = self.n_feats * self.n_window
        self.generator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
        )
        self.discriminator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, 1), nn.Sigmoid(),
        )

    def forward(self, g):
        ## Generate
        z = self.generator(g.view(1, -1))
        ## Discriminator
        real_score = self.discriminator(g.view(1, -1))
        fake_score = self.discriminator(z.view(1, -1))
        return z.view(-1), real_score.view(-1), fake_score.view(-1)


# Proposed Model + Self Conditioning + Adversarial + MAML (VLDB 22)
class TranAD(nn.Module):
    def __init__(self, feats):
        super(TranAD, self).__init__()
        self.name = 'TranAD'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src, tgt):
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2  # 广播机制
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2


class FeatureLayer(nn.Module):
    def __init__(self, device):
        super(FeatureLayer, self).__init__()
        self.device = device

    def forward(self, x):
        matrix_all = []
        y = x.data.cpu().numpy()
        for k in range(y.shape[0]):
            data = y[k]
            matrix = np.dot(data.T, data)
            matrix = matrix / data.shape[0]
            matrix_all.append(matrix)
        attention = torch.from_numpy(np.array(matrix_all))
        attention = attention.to(dtype=torch.double)
        attention = attention.to(self.device)
        x = x.to(torch.device(args.Device))
        h = torch.sigmoid(torch.matmul(attention, x.permute(0, 2, 1)))
        return h.permute(0, 2, 1)


class VARModel(nn.Module):
    def __init__(self, num_variables, order):
        super(VARModel, self).__init__()
        self.num_variables = num_variables
        self.order = order
        self.linear = nn.Linear(num_variables * order, num_variables)

    def forward(self, input_data):
        # Reshape input_data into a suitable format for VAR modeling
        batch_size, num_vars, seq_length = input_data.size()
        input_data = input_data.permute(0, 2, 1).contiguous()
        input_data = input_data.view(batch_size, num_vars, -1)

        # Initialize output tensor
        output = torch.zeros(batch_size, num_vars, seq_length).to(torch.device(args.Device))

        # Perform VAR modeling for each time step
        for t in range(self.order, seq_length):
            input_slice = input_data[:, :, t - self.order:t]
            output_slice = self.linear(input_slice.view(batch_size, -1))
            output[:, :, t] = output_slice
        return output


class Forecasting_Model(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(Forecasting_Model, self).__init__()
        layers = [nn.Linear(in_dim, hid_dim)]

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hid_dim, hid_dim))

        layers.append(nn.Linear(hid_dim, out_dim))

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.relu(self.layers[i](x))
            x = self.dropout(x)

        return self.layers[-1](x)


class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        if channels / reduction < 1:
            a = math.ceil(channels / reduction)
        else:
            a = math.ceil(channels // reduction)
        self.fc1 = nn.Conv1d(channels, a, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(a, channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        # x = torch.softmax(x, dim=1)

        score = x.detach().cpu().numpy()
        np.save("score", score)
        return input * x


class GRULayer(nn.Module):
    """Gated Recurrent Unit (GRU) Layer
    :param in_dim: number of input features
    :param hid_dim: hidden size of the GRU
    :param n_layers: number of layers in GRU
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(GRULayer, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.gru = nn.GRU(in_dim, hid_dim, num_layers=n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):
        out, h = self.gru(x)
        out, h = out[-1, :, :], h[-1, :, :]  # Extracting from last layer
        return out, h


class GATModel(nn.Module):
    def __init__(self, in_features, out_features, num_heads):
        super(GATModel, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_heads):
            self.layers.append(dgl.nn.GATConv(in_features, out_features, num_heads=1))
        self.fc = nn.Linear(out_features, 10)

    def forward(self, g, features):
        all_head_outputs = []
        for gat_layer in self.layers:
            head_output = gat_layer(g, features)
            all_head_outputs.append(head_output)
        output = torch.cat(all_head_outputs, dim=1)
        output = self.fc(output)
        return output


# Proposed Model + Tcn_Local + Tcn_Global + Callback + Transformer + MAML
class DTAAD(nn.Module):
    def __init__(self, feats):
        super(DTAAD, self).__init__()
        self.name = 'DTAAD'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.l_tcn = Tcn_Local(num_outputs=feats, kernel_size=4, dropout=0.2)  # K=3&4 (Batch, output_channel, seq_len)
        self.g_tcn = Tcn_Global(num_inputs=self.n_window, num_outputs=feats, kernel_size=3, dropout=0.2)
        self.pos_encoder = PositionalEncoding(feats, 0.1, self.n_window)
        encoder_layers1 = TransformerEncoderLayer(d_model=feats, nhead=feats, dim_feedforward=16,
                                                  dropout=0.1)  # (seq_len, Batch, output_channel)
        encoder_layers2 = TransformerEncoderLayer(d_model=feats, nhead=feats, dim_feedforward=16,
                                                  dropout=0.1)
        self.transformer_encoder1 = TransformerEncoder(encoder_layers1, num_layers=1)  # only one layer
        self.transformer_encoder2 = TransformerEncoder(encoder_layers2, num_layers=1)
        self.fcn = nn.Linear(feats, feats)
        self.decoder1 = nn.Sequential(nn.Linear(self.n_window, 1), nn.Sigmoid())
        self.decoder2 = nn.Sequential(nn.Linear(self.n_window, 1), nn.Sigmoid())

    def callback(self, src, c):
        src2 = src + c
        g_atts = self.g_tcn(src2)
        src2 = g_atts.permute(2, 0, 1) * math.sqrt(self.n_feats)
        src2 = self.pos_encoder(src2)
        memory = self.transformer_encoder2(src2)
        return memory

    def forward(self, src):
        l_atts = self.l_tcn(src)
        src1 = l_atts.permute(2, 0, 1) * math.sqrt(self.n_feats)
        src1 = self.pos_encoder(src1)
        z1 = self.transformer_encoder1(src1)
        c1 = z1 + self.fcn(z1)
        x1 = self.decoder1(c1.permute(1, 2, 0))
        z2 = self.fcn(self.callback(src, x1))
        c2 = z2 + self.fcn(z2)
        x2 = self.decoder2(c2.permute(1, 2, 0))
        return x1.permute(0, 2, 1), x2.permute(0, 2, 1)  # (Batch, 1, output_channel)


# class PUC(nn.Module):
#     def __init__(self, feats):
#         super(PUC, self).__init__()
#         self.name = 'PUC'
#         self.lr = lr
#         self.batch = 128
#         self.n_feats = feats
#         self.n_window = 10
#         self.g_tcn = Tcn_Global(num_inputs=self.n_window, num_outputs=feats, kernel_size=3, dropout=0.2)
#         self.pos_encoder = PositionalEncoding(feats, 0.2, self.n_window)
#         encoder_layers2 = TransformerEncoderLayer(d_model= feats, nhead=feats, dim_feedforward=16,
#                                                   dropout=0.1)
#         self.transformer_encoder2 = TransformerEncoder(encoder_layers2, num_layers=1)
#         self.se = SEModule(self.n_feats)
#         self.gatModel = GATModel(self.n_feats, self.n_feats, self.n_feats)
#         self.feature = FeatureLayer(args.Device)
#         self.var = VARModel(self.n_feats, order=1)
#         self.forecasting_model = Forecasting_Model(in_dim=self.n_feats, hid_dim=150, out_dim=self.n_feats,
#                                                n_layers=1, dropout=0.1)
#         self.fcn = nn.Linear(feats, feats)
#         self.decoder2 = nn.Sequential(nn.Linear(self.n_window, 1), nn.Sigmoid())
#
#     def forward(self, src):
#         # print('////////////////////////////////////////////////////////////////////')
#         # print('////////////////////////////////////////////////////////////////////')
#         # print('////////////////////////////////////////////////////////////////////')
#         # print('////////////////////////////////////////////////////////////////////')
#         # print(self.n_feats)
#         g_atts = self.g_tcn(src)
#         g_atts = self.se(g_atts)
#         num_nodes = g_atts.size(0)
#         g = dgl.DGLGraph()
#         g = g.to('cuda')
#         g.add_nodes(num_nodes)
#         # 更健壮的边构建方式
#         if num_nodes > 1:
#             # 创建连续边 (i -> i+1)
#             src_nodes = torch.arange(num_nodes - 1, device=src.device)
#             dst_nodes = torch.arange(1, num_nodes, device=src.device)
#             g.add_edges(src_nodes, dst_nodes)
#
#         # 确保至少有一个连接（自环）
#         g.add_edges(torch.arange(num_nodes, device=src.device),
#                     torch.arange(num_nodes, device=src.device))
#
#         # 其余处理保持不变
#         data_processed = torch.mean(g_atts, dim=-1, keepdim=True)
#         features = data_processed.view(num_nodes, -1)
#         getmodelout = self.gatModel(g, features)
#         feature = self.feature(g_atts)
#         nsrc4 = getmodelout + feature + g_atts
#         src2 = nsrc4.permute(2, 0, 1) * math.sqrt(self.n_feats)
#         src2 = self.pos_encoder(src2)
#         memory = self.transformer_encoder2(src2)
#         c2 = memory + self.fcn(memory)
#         x2 = self.decoder2(c2.permute(1, 2, 0))
#         x2f = self.forecasting_model(x2.permute(0, 2, 1))
#         var = self.var(src)
#         x2f = x2f.permute(0, 2, 1) * 0.7 + var * 0.3
#         return x2f.permute(0, 2, 1)

class PUC(nn.Module):
    def __init__(self, feats, dataset, save_path, feature_names):
        super(PUC, self).__init__()
        self.name = 'PUC'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.dataset = dataset
        self.save_path = save_path
        self.feature_names = feature_names
        self.n_window = 10

        # 模型原有组件
        self.g_tcn = Tcn_Global(num_inputs=self.n_window, num_outputs=feats, kernel_size=3, dropout=0.2)
        self.pos_encoder = PositionalEncoding(feats, 0.2, self.n_window)
        encoder_layers2 = TransformerEncoderLayer(d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder2 = TransformerEncoder(encoder_layers2, num_layers=1)
        self.se = SEModule(self.n_feats)
        self.gatModel = GATModel(self.n_feats, self.n_feats, self.n_feats)
        self.feature = FeatureLayer(args.Device)
        self.var = VARModel(self.n_feats, order=1)
        self.forecasting_model = Forecasting_Model(in_dim=self.n_feats, hid_dim=150, out_dim=self.n_feats,
                                                   n_layers=1, dropout=0.1)
        self.fcn = nn.Linear(feats, feats)
        self.decoder2 = nn.Sequential(nn.Linear(self.n_window, 1), nn.Sigmoid())

    def forward(self, src):
        g_atts = self.g_tcn(src)
        g_atts = self.se(g_atts)
        num_nodes = g_atts.size(0)

        # 构建图结构
        g = dgl.DGLGraph()
        g = g.to('cuda')
        g.add_nodes(num_nodes)
        if num_nodes > 1:
            src_nodes = torch.arange(num_nodes - 1, device=src.device)
            dst_nodes = torch.arange(1, num_nodes, device=src.device)
            g.add_edges(src_nodes, dst_nodes)
        g.add_edges(torch.arange(num_nodes, device=src.device),
                    torch.arange(num_nodes, device=src.device))

        data_processed = torch.mean(g_atts, dim=-1, keepdim=True)
        features = data_processed.view(num_nodes, -1)
        getmodelout = self.gatModel(g, features)
        feature = self.feature(g_atts)
        nsrc4 = getmodelout + feature + g_atts
        # nsrc4 = g_atts
        src2 = nsrc4.permute(2, 0, 1) * math.sqrt(self.n_feats)
        src2 = self.pos_encoder(src2)
        memory = self.transformer_encoder2(src2)
        c2 = memory + self.fcn(memory)
        x2 = self.decoder2(c2.permute(1, 2, 0))
        x2f = self.forecasting_model(x2.permute(0, 2, 1))
        var = self.var(src)
        x2f = x2f.permute(0, 2, 1) * 0.7 + var * 0.3
        return x2f.permute(0, 2, 1)


    def plot_dimension_heatmap(self, ascore, timestamps=None):
        """集成时空异常热力图"""
        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'font.size': 18,
            'axes.titlesize': 20,
            'axes.labelsize': 18,
            'xtick.labelsize': 18,
            'ytick.labelsize': 13
        })

        plt.figure(figsize=(18, 8))

        if timestamps is None:
            timestamps = np.arange(ascore.shape[0])

        num_labels = 15
        stride = max(1, len(timestamps) // num_labels)
        indices = np.arange(0, len(timestamps), stride)
        labels = timestamps[::stride]

        ax = sns.heatmap(
            ascore.T,
            xticklabels=False,
            yticklabels=self.feature_names,
            cmap="coolwarm",
            cbar_kws={'label': 'Anomaly Intensity'}
        )

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=18)
        cbar.set_label('Anomaly Intensity', fontsize=18)

        ax.set_title("Multivariate Anomaly Heatmap")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Features")

        ax.set_xticks(indices + 0.5)
        ax.set_xticklabels(labels, rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(f"{self.save_path}{self.dataset}_dimension_heatmap.svg")
        plt.savefig(f"{self.save_path}{self.dataset}_dimension_heatmap.png")
        plt.close()