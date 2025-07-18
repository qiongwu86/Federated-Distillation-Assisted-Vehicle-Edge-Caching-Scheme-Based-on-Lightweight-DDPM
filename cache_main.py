from itertools import chain
import torch
import copy
import numpy as np
from model_ddpm import LightweightUNet1D, GaussianMultinomialDiffusion
from torch.utils.data import DataLoader, TensorDataset
from data_set import convert
from model_ae import DeEncoder, train_autoencoder1
from dataset_processing import sampling_mobility
from cache_options import args_parser
from cache import FedCache_standalone_API

args = args_parser()
sample, users_group_train, users_group_test, users_group_pre, user_request_num = sampling_mobility(args,args.clients_num)
client_models = [
    GaussianMultinomialDiffusion(args.in_out_dim, LightweightUNet1D(), args.num_step, "cpu")
    for _ in range(args.clients_num)]
api = FedCache_standalone_API(client_models, args)
gl_ae = DeEncoder(input_dim=3952, hidden_dim=100, latent_dim=args.in_out_dim)
train_pre = [x for sublist in users_group_pre.values() for x in sublist]
pre_train = convert(sample[train_pre], int(max(sample[:, 1])))
pre_train_data = torch.Tensor(pre_train).float()
print(f"---------------------------- Pre-Training Start------------------------------")
ae_dataloader = DataLoader(pre_train_data[:, 1:-3], batch_size=args.batch_size, shuffle=True)
train_ae = train_autoencoder1(gl_ae, args, ae_dataloader, device=torch.device('cpu'))
ae_wight = [train_ae.state_dict() for _ in range(args.clients_num)]
test_idx = []
for i in range(args.clients_num):
    test_idx.append(users_group_test[i])
test_idx = list(chain.from_iterable(test_idx))
test_dataset = sample[test_idx]
train_data_total = []
for i in range(args.clients_num):
    num = np.random.randint(args.low, args.high)
    train_idx = users_group_train[i][:num]
    train_data = convert(sample[train_idx], int(max(sample[:, 1])))
    train_data = torch.Tensor(train_data).float()
    train_data_total.append(train_data)
api.do_fedcache_stand_alone(train_data_total, args, test_dataset, train_ae, ae_wight)





