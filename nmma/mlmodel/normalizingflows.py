# general modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.lines as mlines
import os, sys, time, glob
import json
import warnings

# pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# nflows
from nflows.nn.nets.resnet import ResidualNet
from nflows import transforms, distributions, flows
from nflows.distributions import StandardNormal
from nflows.flows import Flow
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms import CompositeTransform, RandomPermutation
import nflows.utils as torchutils

# functions
from .embedding import SimilarityEmbedding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_dim = 7
context_features = num_dim
similarity_embedding = SimilarityEmbedding()
num_points = 121


class EmbeddingNet(nn.Module):
    """Wrapper around the similarity embedding defined above"""

    # def __init__(self, *args, **kwargs):
    def __init__(self, similarity_embedding, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.representation_net = similarity_embedding
        self.representation_net.load_state_dict(similarity_embedding.state_dict())
        # self.num_dim = num_dim
        # the expander network is unused and hence don't track gradients
        for name, param in self.representation_net.named_parameters():
            if "expander_layer" in name or "layers_h" in name or "final_layer" in name:
                param.requires_grad = False
            # set freeze status of part of the conv layer of embedding_net
            elif "layers_f" in name:
                param.requires_grad = True
            else:
                param.requires_grad = True
        self.context_layer = nn.Sequential(
            nn.Linear(num_dim, 1000), nn.ReLU(), nn.Linear(1000, num_dim)
        )

    def forward(self, x):
        batch_size, channels, dims = x.shape
        _, rep = self.representation_net(x)  # 500, 3, 191 -> 500, 1, 2
        rep = rep.reshape(batch_size, num_dim)

        return self.context_layer(rep)


def normflow_params(
    similarity_embedding,
    num_transforms,
    num_blocks,
    hidden_features,
    context_features,
    num_dim,
):
    base_dist = StandardNormal([3])
    transforms = []
    features = 3
    for _ in range(num_transforms):
        block = [
            MaskedAffineAutoregressiveTransform(
                features=3,
                hidden_features=hidden_features,  # 80
                context_features=context_features,  # 5
                num_blocks=num_blocks,  # 5
                activation=torch.tanh,
                use_batch_norm=False,
                use_residual_blocks=True,
                dropout_probability=0.01,
                #             integrand_net_layers=[20, 20]
            ),
            RandomPermutation(features=features),
        ]
        transforms += block
    transform = CompositeTransform(transforms)
    embedding_net = EmbeddingNet(similarity_embedding)
    return transform, base_dist, embedding_net


def train_one_epoch(epoch_index, tb_writer, data_loader, flow, optimizer, flatten_dim):
    running_loss = 0.0
    last_loss = 0.0
    for idx, val in enumerate(data_loader, 1):
        augmented_shift, unshifted_shift, augmented_data, unshifted_data = val
        augmented_shift = augmented_shift[..., 0:3].to(device)
        augmented_shift = augmented_shift.flatten(0, flatten_dim).to(device)
        augmented_data = augmented_data.reshape(-1, 3, num_points).to(device)
        loss = 0
        flow_loss = -flow.log_prob(augmented_shift, context=augmented_data).mean()
        optimizer.zero_grad()
        flow_loss.backward()
        optimizer.step()
        loss += flow_loss.item()
        running_loss += loss
        n = 10
        if idx % n == 0:
            last_loss = running_loss / n
            print(
                " Avg. train loss/batch after {} batches = {:.4f}".format(
                    idx, last_loss
                )
            )
            tb_x = epoch_index * len(data_loader) + idx
            tb_writer.add_scalar("Flow Loss/train", last_loss, tb_x)
            tb_writer.flush()
            running_loss = 0.0
    return last_loss


def val_one_epoch(epoch_index, tb_writer, data_loader, flow, flatten_dim):
    running_loss = 0.0
    last_loss = 0.0
    for idx, val in enumerate(data_loader, 1):
        augmented_shift, unshifted_shift, augmented_data, unshifted_data = val
        augmented_shift = augmented_shift[..., 0:3].to(device)
        augmented_shift = augmented_shift.flatten(0, flatten_dim).to(device)
        augmented_data = augmented_data.reshape(-1, 3, num_points).to(device)
        loss = 0
        flow_loss = -flow.log_prob(augmented_shift, context=augmented_data).mean()
        loss += flow_loss.item()
        running_loss += loss
        n = 1
        if idx % n == 0:
            last_loss = running_loss / n
            print(
                " Avg. train loss/batch after {} batches = {:.4f}".format(
                    idx, last_loss
                )
            )
            tb_x = epoch_index * len(data_loader) + idx + 1
            tb_writer.add_scalar("Flow Loss/val", last_loss, tb_x)
            tb_writer.flush()
            running_loss = 0.0
    tb_writer.flush()
    return last_loss
