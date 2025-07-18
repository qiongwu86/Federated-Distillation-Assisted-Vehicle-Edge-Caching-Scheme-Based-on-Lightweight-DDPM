import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments (Notation for the arguments followed from paper)
    # DDPM 参数
    parser.add_argument('--comm_round', type=int, default=30,
                        help="number of training: S")
    parser.add_argument('--clients_num', type=int, default=20,
                        help="number of rounds of training")
    parser.add_argument('--R', type=int, default=110,
                        help="the number of local epochs: E")
    parser.add_argument('--loc_ep', type=int, default=20,
                        help="the number of local epochs: E")
    parser.add_argument('--dm_lr', type=int, default=0.0006,
                        help="number of rounds of training")
    parser.add_argument('--num_step', type=int, default=50,
                        help='learning rate')
    parser.add_argument('--T', type=float, default=1.0,
                        help="the number of local epochs: E")
    parser.add_argument('--alpha', type=float, default=1.0,
                        help="the number of local epochs: E")
    parser.add_argument('--sim_num', type=int, default=5,
                        help="the number of local epochs: E")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="the number of local epochs: E")
    parser.add_argument('--in_out_dim', type=int, default=16,
                        help="the number of local epochs: E")
    parser.add_argument('--pre_ae_ep', type=int, default=30,
                        help="the number of local epochs: E")
    parser.add_argument('--ae_lr', type=int, default=0.02,
                        help="the number of local epochs: E")
    parser.add_argument('--loc_ae_ep', type=int, default=30,
                        help="the number of local epochs: E")
    parser.add_argument('--low', type=int, default=15000,
                        help="the number of local epochs: E")
    parser.add_argument('--high', type=int, default=20000,
                        help="the number of local epochs: E")

    parser.add_argument('--clean_dataset', type=bool, default=True, help="clean\
                             the model/data_set or not")
    parser.add_argument('--clean_user', type=bool, default=True, help="clean\
                             the user/ or not")
    parser.add_argument('--clean_clients', type=bool, default=True, help="clean\
                             the model/clients or not")
    parser.add_argument('--clean_clients_density', type=bool, default=True, help="clean\
                             the model/clients or not")
    parser.add_argument('--dataset', type=str, default='ml-1m', help="name of dataset")
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                             to a specific GPU ID. Default set to use CPU.")
    args = parser.parse_args()
    return args