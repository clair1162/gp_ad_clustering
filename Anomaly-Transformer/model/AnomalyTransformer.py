import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import AnomalyAttention, AttentionLayer
from .embed import DataEmbedding, TokenEmbedding

import scipy
import scipy.cluster.hierarchy as sch
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# %matplotlib inline
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from sklearn.manifold import TSNE
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# x.detach().cpu().numpy()
# np.array(x)
# sns.heatmap(cluster_corr(df.corr()))



class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model

        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

        # print("attention : ", self.attention, "conv1 : ", self.conv1, "conv2 : ", self.conv2, "norm1 : ", self.norm1, "norm2 : ", self.norm2, " dropout : ", self.dropout, sep=" | ")

    def forward(self, x, attn_mask=None):
        new_x, attn, mask, sigma = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))


        return self.norm2(x + y), attn, mask, sigma


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D] [256, 100, 512]
        series_list = [] # [[256, 8, 100, 100]]
        prior_list = []
        sigma_list = []

     
        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            # 여기다가 넣는게 좋지 않을까 하구.. 따로 분리하는게 나은가..? 따로 분리한 값도 있긴하지만서두. 흠. ********************************************************************
            # print("*****series : ", series.shape)  256*8*100*100
            # series_cluster = series
            # series_cluster = series_cluster.detach().cpu().numpy()
            # np.transpose(series_cluster, (0,1,3,2)) 근데 이때 이걸 왜 바꿨지?? ?? 진짜 왜 바꾼거임

            # x is my data which is a nd-array
            # You have to convert your tensor to nd-array before using scikit-learn's tsne
            # Convert your tensor to x =====> x = tf.Session().run(tensor_x)
            # series 말고 x 어떨까..
            x_cluster = series
            x_cluster = x_cluster.detach().cpu().numpy()
            standard = StandardScaler()
            kmeans = KMeans(x_cluster) # 옵션들은 찾아보고..
            # loss에 참여시킬것 아니면 값은 딱히 됐고.. 

           


            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list


class AnomalyTransformer(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True):
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention

        # Encoding
        self.embedding = DataEmbedding(enc_in, d_model, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        enc_out = self.embedding(x)
        enc_out, series, prior, sigmas = self.encoder(enc_out)
        enc_out = self.projection(enc_out)



        if self.output_attention:
            return enc_out, series, prior, sigmas
        else:
            return enc_out  # [B, L, D]
