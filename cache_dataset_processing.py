from collections import Counter
import torch
from dataset_processing import cache_efficiency4
from model_ae import generator_data
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import scipy.stats as stats

def top_K_numbers(sim_lab, K):
    all_numbers = []
    for arr in sim_lab:
        all_numbers.extend(arr.tolist())  # 或 arr.flatten().tolist()
    count = Counter(all_numbers)
    top_3_numbers, top_3_counts = zip(*count.most_common(K))  # 解包为数字和次数

    return top_3_numbers


def cache_hit_ratio(generated_data, test_dataset, k):

    generated_ratings = torch.sum(generated_data, dim=0)

    return cache_efficiency4(generated_ratings, test_dataset, k)



def get_user_cluster(user_id, user_hash, cache, top_k):


    all_uids = np.array(list(cache.keys()), dtype=np.uint16)
    all_vecs = np.vstack(list(cache.values()))
    mask = (all_uids != user_id)
    filt_uids = all_uids[mask]
    filt_vecs = all_vecs[mask]
    if filt_uids.size == 0:
        return np.zeros(0, dtype=np.uint16)
    if user_hash.ndim == 1:
        user_hash = user_hash.reshape(1, -1)
    sims = cosine_similarity(filt_vecs, user_hash).flatten()
    top_idxs = np.argsort(-sims)[:top_k]
    user_sim = filt_uids[top_idxs]
    return user_sim


def knowledge_avg_single(knowledge, weights):
    result = torch.zeros_like(knowledge[0]).cpu()
    total_w = 0.0
    w = 1
    for vec, _ in zip(knowledge, weights):
        result.add_(vec.cpu() * w)
        total_w += w
    result = result / total_w
    return result


def vehicle_mobility(client_num):
    vehicle_dis=np.zeros(client_num)
    mu, sigma = 38,1
    lower, upper = mu - 2 * sigma, mu + 2 * sigma
    x=stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    veh_speed=x.rvs(client_num)
    for i in range(len(veh_speed)):
        veh_speed[i] = veh_speed[i] * 0.278
    print("each vehicle's speed:",veh_speed,'m/s')
    return veh_speed, vehicle_dis

def vehicle_mobility_updata(veh_dis,epoch_time):
    mu, sigma = 38, 1
    lower, upper = mu - 2 * sigma, mu + 2 * sigma
    x = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    veh_speed = x.rvs(len(veh_dis))
    for i in range(len(veh_speed)):
        veh_speed[i] = veh_speed[i] * 0.278
    for i in range(len(veh_dis)):
        veh_dis[i]+=veh_speed[i]*epoch_time[i]
    return veh_dis,veh_speed

def request_delay(cache_hit_ratios, v2i_rate):

    request_num = 700
    comm_rate =  v2i_rate
    v2i_rate_mbs = 0.4 * v2i_rate
    request_delay_all = []
    for j in range(len(cache_hit_ratios)):
        cache_hit_ratio = cache_hit_ratios[j] / 100
        request_delay = 0
        request_delay += cache_hit_ratio * (request_num / comm_rate) * 800000
        request_delay += (1 - cache_hit_ratio) * (request_num / v2i_rate_mbs) * 800000
        request_delay_all.append(request_delay)
    return request_delay_all

def cache_efficiency_all(generated_ratings,test_dataset):
    generated_ratings = torch.sum(generated_ratings, dim=0)
    k_list = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    CACHE_HIT_RATIO = []
    Oracle_hit_ratio = []
    for i in range(len(k_list)):
        k = k_list[i]
        top_k_values, top_k_indices = torch.topk(generated_ratings, k)
        top_k_indices += 1
        top_movie_indices = top_k_indices.numpy()
        requset_items = test_dataset[:, 1]
        count = Counter(requset_items)
        CACHE_HIT_NUM = 0
        for item in top_movie_indices:
            CACHE_HIT_NUM = CACHE_HIT_NUM + count[item]
        CACHE_HIT_RATIO.append(CACHE_HIT_NUM / len(requset_items) * 100)

        Oracle_hit = sum(list(sorted(count.values()))[-k:])
        Oracle_hit_ratio.append(Oracle_hit / len(requset_items) * 100)

    return CACHE_HIT_RATIO, Oracle_hit_ratio



if __name__ == '__main__':
    veh_speed, vehicle_dis = vehicle_mobility(5)
    print("veh_speed, vehicle_dis:",veh_speed, vehicle_dis)