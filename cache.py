import torch
import time
import cache_utils  # 工具模块，提供 KL 损失、平均计算、准确率指标等
from torch.nn import functional as F  # PyTorch 函数式接口，用于计算损失
import numpy as np
import torch.nn as nn
from itertools import chain
from torch.utils.data import DataLoader
from env_comm import V2Ichannels, Environ
from model_ae import generator_data, train_autoencoder, DeEncoder
from cache_dataset_processing import vehicle_mobility,vehicle_mobility_updata,request_delay,top_K_numbers
from cache_dataset_processing import cache_hit_ratio, get_user_cluster, knowledge_avg_single, cache_efficiency_all

np.random.seed(0)

class FedCache_standalone_API:
    def __init__(self, client_models,args):
        self.client_models = client_models

        self.criterion_KL = cache_utils.KL_Loss()
        self.criterion_CE = F.cross_entropy
        self.hash_encoder = nn.Sequential(
            nn.Linear(3952, args.in_out_dim),
            nn.Sigmoid()
        ).to('cpu')
        self.cache = {}
        self.cli_sim_lab = {i: {} for i in range(args.clients_num)}
        self.logit_cache = {}
    def do_fedcache_stand_alone(self, train_data, args, test_dataset, ae, ae_wight):

        veh_speed, veh_dis = vehicle_mobility(args.clients_num)
        env = Environ(args.clients_num)
        env.new_random_game(veh_dis, veh_speed)
        train_loaders = []
        for i in range(args.clients_num):
            ae.eval()
            with torch.no_grad():
                data_dim = ae.encoder(train_data[i][:, 1:-3])
            train_loaders.append(DataLoader(data_dim, batch_size=args.batch_size, shuffle=True))
            size_bytes = data_dim.storage().nbytes()
        for cli in range(args.clients_num):
            for data_point in train_loaders[cli]:
                hashes = data_point
                for vec in hashes:
                    self.cache.update({cli: vec})
        for cli in range(args.clients_num):
            sim_lab = []
            for data_point in train_loaders[cli]:
                for vec in data_point:
                    sim = get_user_cluster(int(cli), vec, self.cache, args.sim_num)
                    sim_lab.append(sim)
            self.cli_sim_lab[cli] = top_K_numbers(sim_lab,args.sim_num)

        ae_loc = DeEncoder(input_dim=3952, hidden_dim=100, latent_dim=args.in_out_dim)
        cache_hit_total = []
        loss_all = []
        for round_idx in range(args.comm_round):
            print(f"--- 第 {round_idx+1} 轮通信 ---")
            epoch_time_all = []
            v2i_rate = env.Compute_Performance_Train_mobility()
            generated_data_all = []
            cache_hit_500 = []
            for cid, model in enumerate(self.client_models):
                model.train()
                optimizer = torch.optim.Adam(model.parameters(), lr=args.dm_lr)
                total_loss = 0
                g = 0
                ae_loc.load_state_dict(ae_wight[cid])
                ae_dataloader = DataLoader(train_data[cid][:, 1:-3], batch_size=args.batch_size, shuffle=True)
                tra_ae_loc = train_autoencoder(ae_loc, args, ae_dataloader,round_idx, device=torch.device('cpu'))
                tra_ae_loc.eval()
                with torch.no_grad():
                    data_dm = tra_ae_loc.encoder(train_data[cid][:, 1:-3])
                ae_wight[cid] = tra_ae_loc.state_dict()
                dm_dataloader = DataLoader(data_dm, batch_size=args.batch_size, shuffle=True)
                start_time = time.time()
                for idx in range(args.loc_ep):
                    for data_point in dm_dataloader:
                        t = torch.randint(0, args.num_step, (data_point.size(0),))
                        loss_mse, logits = model.compute_loss1(data_point, t)
                        if g==0:
                            self.logit_cache.update({cid: logits.detach()})
                            size_bytes = logits.storage().nbytes()
                            g = 1
                        fetched = [self.logit_cache[key] for key in self.cli_sim_lab[cid] if key in self.logit_cache]
                        if not fetched:
                            batch_size, num_classes = logits.shape  # 假设 logits 形状为 [batch_size, num_classes]
                            teacher_avg = torch.zeros(batch_size, num_classes) # 零向量作为平均结果
                            loss_kd = torch.tensor(0.0)
                        else:
                            teacher_avg = torch.stack([
                                knowledge_avg_single(neighbors, [1]*args.R)
                                for neighbors in fetched
                            ])
                            mean_vec = teacher_avg.mean(dim=0)
                            logits = logits.mean(dim=0)
                            loss_kd = self.criterion_KL(logits, mean_vec / args.T)
                        loss = loss_mse + args.alpha * loss_kd
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                    if cid == 0:
                        loss_all.append(loss.item())
                avg_loss = total_loss / (len(train_loaders[cid])+args.loc_ep)
                print(f"- User {cid+1} ---- DDPM Training ----- Local Epoch {args.loc_ep}- Loss: {avg_loss:.4f}")
                if cid == 0:
                    print("每次迭代的loss:" ,loss_all)
                epoch_time = time.time() - start_time
                epoch_time_all.append(epoch_time)
                # 缓存命中计算
                generated_data = generator_data(model, 100)
                tra_ae_loc.eval()
                with torch.no_grad():
                    generated_data = tra_ae_loc.decoder(generated_data)
                cache_hit_500.append(cache_hit_ratio(generated_data, test_dataset, 500))
                w = (1000-veh_dis[cid])/veh_speed[cid]
                generated_data_all.append(generated_data*w)
            generated_data_all = torch.stack(list(chain.from_iterable(generated_data_all)), dim=0)
            cache_hit_all = cache_hit_ratio(generated_data_all, test_dataset, 500)
            cache_hit_total.append(cache_hit_all)
            veh_dis, veh_speed = vehicle_mobility_updata(veh_dis, epoch_time_all)
            env.new_random_game(veh_dis, veh_speed)
        print('cache_hit_total:',cache_hit_total)
        print("FedCache 训练完成")


