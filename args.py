import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="cora")
current_path = os.path.abspath(__file__)
dataset_path = os.path.join(os.path.dirname(current_path), 'datasets')
root_dir = os.path.join(dataset_path, 'raw_data')
if not os.path.exists(root_dir):
    os.makedirs(root_dir)
parser.add_argument("--dataset_dir", type=str, default=root_dir)

log_path = os.path.join(os.path.dirname(current_path), 'logs')
if not os.path.exists(log_path):
    os.makedirs(log_path)

parser.add_argument("--logs_dir", type=str, default=log_path)
parser.add_argument("--specified_domain_skew_task", type=str, default=None)
parser.add_argument("--task", type=str, default="node_classification")
parser.add_argument("--skew_type", type=str, default="label_skew")
parser.add_argument("--train_val_test_split", type=list, default=[0.2, 0.4, 0.4])
parser.add_argument("--dataset_split_metric", type=str, default="transductive")

parser.add_argument("--num_rounds", type=int, default=1)
parser.add_argument("--num_clients", type=int, default=10)
parser.add_argument("--T_L", type=int, default=100)
parser.add_argument("--cl_sample_rate", type=float, default=1.0)
parser.add_argument("--evaluation_mode", type=str, default="global")
parser.add_argument("--fed_algorithm", type=str, default="GHOST")
parser.add_argument("--model", type=str, default="GCN")
parser.add_argument("--hidden_dim", type=int, default=128)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--f_scale", type=float, default=1)
parser.add_argument("--r_scale", type=float, default=1e-11)
parser.add_argument("--n_scale", type=float, default=1e-11)
parser.add_argument("--learning_rate", type=float, default=0.005)
parser.add_argument("--weight_decay", type=float, default=4e-4)

parser.add_argument("--noise_dim", type=int, default=64)
parser.add_argument("--M", type=int, default=3)
parser.add_argument("--T_G", type=int, default=5)

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device_id", type=int, default=0)
parser.add_argument("--dirichlet_alpha", type=float, default=0.05)
parser.add_argument("--least_samples", type=int, default=5)
parser.add_argument("--dirichlet_try_cnt", type=int, default=10000)


#===============================================MOSAIC==============================


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value from: {v}")


parser.add_argument("--sf_local_epochs", type=int, default=100)
parser.add_argument("--sf_eval_every", type=int, default=5)
parser.add_argument("--sf_use_best_ckpt", type=str2bool, default=True)


parser.add_argument("--mosaic_server_epochs", type=int, default=200)
parser.add_argument("--mosaic_server_eval_every", type=int, default=10)
parser.add_argument("--mosaic_server_use_best_ckpt", type=str2bool, default=True)
parser.add_argument("--mosaic_server_lr", type=float, default=1e-3)
parser.add_argument("--mosaic_server_wd", type=float, default=5e-4)


parser.add_argument("--mosaic_init_from_experts", type=str2bool, default=True)
parser.add_argument("--mosaic_pseudo_val_ratio", type=float, default=0.2)
parser.add_argument("--mosaic_random_seed", type=int, default=0)
parser.add_argument("--mosaic_edge_per_node", type=int, default=None)
parser.add_argument("--mosaic_transition_edge_ratio", type=float, default=None)
parser.add_argument("--mosaic_knn_k", type=int, default=None)


parser.add_argument("--mosaic_base_per_class", type=int, default=None)
parser.add_argument("--mosaic_total_pseudo_nodes", type=int, default=None)
parser.add_argument("--mosaic_feat_std_scale", type=float, default=1.0)

parser.add_argument("--sf_gate_temp", type=float, default=0.5)

parser.add_argument("--mosaic_kd_coef", type=float, default=1.0)
parser.add_argument("--mosaic_hard_coef", type=float, default=1.0)
parser.add_argument("--mosaic_kd_temp", type=float, default=1.5)
parser.add_argument("--mosaic_ce_beta", type=float, default=1.0)

#===============================================MOSAIC-Ablation==============================

parser.add_argument(
    "--mosaic_atlas_ablation",
    type=str,
    default="none",
    choices=["none", "no_t", "no_f"]
)

parser.add_argument(
    "--mosaic_graph_ablation",
    type=str,
    default="none",
    choices=["none", "transition_only", "knn_only"]
)

parser.add_argument(
    "--mosaic_teacher_ablation",
    type=str,
    default="none",
    choices=["none", "uniform_average", "no_structural"]
)

parser.add_argument(
    "--mosaic_distill_ablation",
    type=str,
    default="full",
    choices=["full", "no_kd", "no_ce", "equal_kd_ce", "fixed_adaptive_sum"]
)

parser.add_argument(
    "--mosaic_fixed_adaptive_sum",
    type=float,
    default=1.0
)


#----------------------Cora----------------------------------
parser.add_argument("--lambda_d", type=float, default=0.5)
parser.add_argument("--lambda_f", type=float, default=0.2)
parser.add_argument("--lambda_r", type=float, default=0.1)
parser.add_argument("--lambda_n", type=float, default=0.1)


#----------------------Citeseer----------------------------------
'''parser.add_argument("--lambda_d", type=float, default=0.1)
parser.add_argument("--lambda_f", type=float, default=0.1)
parser.add_argument("--lambda_r", type=float, default=1.0)
parser.add_argument("--lambda_n", type=float, default=1.0)'''


#----------------------Chameleon----------------------------------
'''parser.add_argument("--lambda_d", type=float, default=0.01)
parser.add_argument("--lambda_f", type=float, default=1)
parser.add_argument("--lambda_r", type=float, default=1.0)
parser.add_argument("--lambda_n", type=float, default=0.5)'''

#----------------------Pubmed----------------------------------
'''parser.add_argument("--lambda_d", type=float, default=0.01)
parser.add_argument("--lambda_f", type=float, default=0.5)
parser.add_argument("--lambda_r", type=float, default=1.0)
parser.add_argument("--lambda_n", type=float, default=1.0)'''

#----------------------Photo----------------------------------
'''parser.add_argument("--lambda_d", type=float, default=0.1)
parser.add_argument("--lambda_f", type=float, default=0.1)
parser.add_argument("--lambda_r", type=float, default=0.5)
parser.add_argument("--lambda_n", type=float, default=0.5)'''

#----------------------OGBN-Arxiv----------------------------------
'''parser.add_argument("--lambda_d", type=float, default=0.5)
parser.add_argument("--lambda_f", type=float, default=0.2)
parser.add_argument("--lambda_r", type=float, default=1.0)
parser.add_argument("--lambda_n", type=float, default=1.0)'''




#----------------------FedNova----------------------------------

parser.add_argument("--fednova_eta", type=float, default=1.0)

parser.add_argument("--feddc_alpha", type=float, default=1.0)

parser.add_argument("--feddyn_alpha", type=float, default=1.0)


#----------------------FedProx----------------------------------

parser.add_argument("--fedprox_mu", type=float, default=0.01)


#----------------------FedFisher----------------------------------

parser.add_argument("--fedfisher_local_epochs", type=int, default=100)
parser.add_argument("--fedfisher_lr", type=float, default=0.005)
parser.add_argument("--fedfisher_momentum", type=float, default=0.9)
parser.add_argument("--fedfisher_weight_decay", type=float, default=4e-4)

parser.add_argument("--fedfisher_fisher_batch_size", type=int, default=256)

parser.add_argument("--fedfisher_server_eta", type=float, default=0.01)
parser.add_argument("--fedfisher_server_T", type=int, default=2000)
parser.add_argument("--fedfisher_server_beta1", type=float, default=0.9)
parser.add_argument("--fedfisher_server_beta2", type=float, default=0.99)
parser.add_argument("--fedfisher_server_eps", type=float, default=0.01)
parser.add_argument("--fedfisher_server_eval_every", type=int, default=100)

parser.add_argument("--fedfisher_quant_bits", type=int, default=15)


#----------------------FedGTA----------------------------------

parser.add_argument("--fedgta_prop_steps", type=int, default=5)
parser.add_argument("--fedgta_lp_alpha", type=float, default=0.5)
parser.add_argument("--fedgta_temperature", type=float, default=20.0)
parser.add_argument("--fedgta_num_moments", type=int, default=10)
fedgta_moment_type_choices = ["raw", "central", "hybrid"]
parser.add_argument("--fedgta_moment_type", type=str, default="hybrid", choices=fedgta_moment_type_choices)
parser.add_argument("--fedgta_accept_alpha", type=float, default=0.5)





#----------------------FGSSL----------------------------------


parser.add_argument("--fgssl_distill_loss_weight", type=float, default=0.5,
                    help="Distillation loss weight for FGSSL")
parser.add_argument("--fgssl_contrastive_loss_weight", type=float, default=0.5,
                    help="Contrastive loss weight for FGSSL")


#----------------------FedTAD----------------------------------

parser.add_argument("--fedtad_noise_dim", type=int, default=32)
parser.add_argument("--fedtad_num_gen", type=int, default=100)
parser.add_argument("--fedtad_glb_epochs", type=int, default=5)
parser.add_argument('--fedtad_it_g', type=int, default=1)
parser.add_argument('--fedtad_it_d', type=int, default=5)
parser.add_argument('--fedtad_topk', type=int, default=5)
parser.add_argument("--fedtad_lam1", type=float, default=1.0)
parser.add_argument("--fedtad_lam2", type=float, default=1.0)
fedtad_distill_mode_choices = ["rep_distill", "raw_distill"]
parser.add_argument("--fedtad_distill_mode", type=str, default="raw_distill", choices=fedtad_distill_mode_choices)


#----------------------FedSD2C----------------------------------

parser.add_argument("--fedsd2c_client_instance", type=str, default="coreset+dist_syn",
                    choices=["coreset", "coreset+dist_syn"])

parser.add_argument("--fedsd2c_ipc", type=int, default=20)
parser.add_argument("--fedsd2c_mipc", type=int, default=10)

parser.add_argument("--fedsd2c_iteration", type=int, default=50)
parser.add_argument("--fedsd2c_lr", type=float, default=5e-3)
parser.add_argument("--fedsd2c_l2_scale", type=float, default=1e-3)
parser.add_argument("--fedsd2c_syn_noise", type=float, default=0.05)

parser.add_argument("--fedsd2c_knn_k", type=int, default=8)

parser.add_argument("--fedsd2c_server_epochs", type=int, default=200)
parser.add_argument("--fedsd2c_server_optimizer", type=str, default="Adam",
                    choices=["SGD", "Adam", "AdamW"])
parser.add_argument("--fedsd2c_server_lr", type=float, default=1e-3)
parser.add_argument("--fedsd2c_server_momentum", type=float, default=0.9)

parser.add_argument("--fedsd2c_temperature", type=float, default=1.0)
parser.add_argument("--fedsd2c_kd_weight", type=float, default=1.0)
parser.add_argument("--fedsd2c_ce_weight", type=float, default=1.0)

parser.add_argument("--fedsd2c_log_every", type=int, default=10)


args = parser.parse_args()

