import argparse
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import json
import numpy as np
from config import config
print('Current working directory:', os.getcwd())
from src.models_dual_inter.model import siMLPe as Model 
from src.models_dual_inter.utils import visuaulize,seed_set,get_dct_matrix,gen_velocity,predict,getRandomPermuteOrder,getRandomRotatePoseTransform#很重要，切换模型时将models_inter修改为其他文件夹
from lr import update_lr_multistep
from src.baseline_h36m_30to30.lib.datasets.dataset_mocap import DATA
from lib.utils.logger import get_logger, print_and_log_info
from lib.utils.pyt_utils import  ensure_dir
import torch
from torch.utils.tensorboard import SummaryWriter
from src.baseline_h36m_30to30.test_mupots import mpjpe_test_regress,regress_pred,mpjpe_vim_test
import shutil

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--exp-name', type=str, default="+(1)", help='Name of the experiment, a folder with this name will be created for each experiment')
parser.add_argument('--dataset', type=str, default="others", help='Unused')
parser.add_argument('--seed', type=int, default=888, help='=seed')#default:888
parser.add_argument('--temporal-only', action='store_true', help='Only transform in the time dimension, unused')
parser.add_argument('--layer-norm-axis', type=str, default='spatial', help='Do LN in D dimension, unused')
parser.add_argument('--with-normalization', type=bool,default=True, help='Whether to use LN, unused')
parser.add_argument('--spatial-fc', action='store_true', help='Unused')
parser.add_argument('--normalization',type=bool,default=True, help='Whether to use normalization, mupots use normalization better, mocap does not use normalization better')
parser.add_argument('--norm_way',type=str,default='first', help='Normalization way, all means subtract the first frame of each person, first means subtract the first frame of the first person')
parser.add_argument('--num', type=int, default=64, help='=num of blocks')
parser.add_argument('--interaction_interval', type=int, default=16, help='Interval between local and Global interactions, must be divisible by num')
parser.add_argument('--random_rotate', type=bool, default=True, help='Random rotation around the world center')
parser.add_argument('--permute_p', type=bool, default=True, help='Permute P dimension')
parser.add_argument('--weight', type=float, default=1., help='=loss weight, unused')
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--debug', type=bool, default=False)#unused
parser.add_argument('--n_p', type=int, default=3)#Number of people in the dataset
parser.add_argument('--model_path', type=str, default=None)#Pre-trained model address
parser.add_argument('--vis_every', type=int, default=2500000000000)#Interval for visualizing once
parser.add_argument('--save_every', type=int, default=250)#Interval for evaluating
parser.add_argument('--print_every', type=int, default=100)#Interval for printing loss
parser.add_argument('--batch_size', type=int, default=128)
args = parser.parse_args()

# Create experiment folder
expr_dir = os.path.join('exprs', args.exp_name)
if os.path.exists(expr_dir):
    shutil.rmtree(expr_dir)
os.makedirs(expr_dir, exist_ok=True)

# Ensure reproducibility of experiments
seed_set(args.seed)
torch.use_deterministic_algorithms(True)



# Configuration
config.norm_way=args.norm_way
config.normalization=args.normalization
config.batch_size = args.batch_size
config.dataset = args.dataset
config.n_p = args.n_p
config.vis_every = args.vis_every
config.save_every = args.save_every
config.print_every = args.print_every
config.debug = args.debug
config.device = args.device
config.expr_dir=expr_dir
config.motion_fc_in.temporal_fc = args.temporal_only
config.motion_fc_out.temporal_fc = args.temporal_only
config.motion_mlp.norm_axis = args.layer_norm_axis
config.motion_mlp.spatial_fc_only = args.spatial_fc
config.motion_mlp.with_normalization = args.with_normalization
config.motion_mlp.num_layers = args.num
config.motion_mlp.n_p = args.n_p
config.motion_mlp.seq_local_D=config.motion_mlp.seq_len
config.motion_mlp.seq_global_D=config.motion_mlp.seq_len*args.n_p
# config.motion_mlp.seq_local_D=40
# config.motion_mlp.seq_global_D=40
config.motion_mlp.interaction_interval = args.interaction_interval
config.snapshot_dir=os.path.join(expr_dir, 'snapshot')
ensure_dir(config.snapshot_dir)# Create folder
config.vis_dir=os.path.join(expr_dir, 'vis')
ensure_dir(config.vis_dir)# Create folder
config.log_file=os.path.join(expr_dir, 'log.txt')
config.model_pth=args.model_path

#
writer = SummaryWriter()

# Get dct matrix
dct_m,idct_m = get_dct_matrix(config.dct_len)
dct_m = torch.tensor(dct_m).float().to(config.device).unsqueeze(0)
idct_m = torch.tensor(idct_m).float().to(config.device).unsqueeze(0)
config.dct_m=dct_m
config.idct_m=idct_m

# Training function
def train_step(h36m_motion_input, h36m_motion_target, model, optimizer, nb_iter, total_iter, max_lr, min_lr) :
    if args.random_rotate:# Data augmentation 1
        h36m_motion_input,h36m_motion_target=getRandomRotatePoseTransform(h36m_motion_input,h36m_motion_target)
    if args.permute_p:# Data augmentation 2
        h36m_motion_input,h36m_motion_target=getRandomPermuteOrder(h36m_motion_input,h36m_motion_target)
    motion_pred=predict(model,h36m_motion_input,config)#b,p,n,c, Predicted pose
    b,p,n,c = h36m_motion_target.shape
    # Predicted pose
    motion_pred = motion_pred.reshape(b,p,n,config.n_joint,3).reshape(-1,3)
    # GT
    h36m_motion_target = h36m_motion_target.to(config.device).reshape(b,p,n,config.n_joint,3).reshape(-1,3)
    # Calculate loss
    loss = torch.mean(torch.norm(motion_pred - h36m_motion_target, 2, 1))

    if  config.use_relative_loss:# Calculate velocity loss
        motion_pred = motion_pred.reshape(b,p,n,config.n_joint,3)
        dmotion_pred = gen_velocity(motion_pred)# Calculate velocity
        
        motion_gt = h36m_motion_target.reshape(b,p,n,config.n_joint,3)
        dmotion_gt = gen_velocity(motion_gt)# Calculate velocity
        
        dloss = torch.mean(torch.norm((dmotion_pred - dmotion_gt).reshape(-1,3), 2, 1))
        loss = loss + dloss
    else:
        loss = loss.mean()

    writer.add_scalar('Loss/angle', loss.detach().cpu().numpy(), nb_iter)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer, current_lr = update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer)
    writer.add_scalar('LR/train', current_lr, nb_iter)

    return loss.item(), optimizer, current_lr

# Create model
model = Model(config).to(device=config.device)
model.train()
print(">>> total params: {:.2f}M".format(
    sum(p.numel() for p in list(model.parameters())) / 1000000.0))


if config.dataset=="h36m":
    pass
else:
    dataset = DATA( 'train', config.t_his,config.t_pred,n_p=config.n_p)
    eval_dataset_mocap = DATA( 'eval_mocap', config.t_his,config.t_pred_eval,n_p=config.n_p)
    eval_dataset_mupots=DATA('eval_mutpots',config.t_his,config.t_pred_eval,n_p=config.n_p)
    
# initialize optimizer
optimizer = torch.optim.Adam(model.parameters(),
                             lr=config.cos_lr_max,
                             weight_decay=config.weight_decay)
# Create logger
def default_serializer(obj):
    if isinstance(obj, torch.Tensor):
        return None  # or return a mark, or convert it to a serializable type
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

logger = get_logger(config.log_file, 'train')
print_and_log_info(logger, json.dumps(config, indent=4, sort_keys=True,default=default_serializer))

if config.model_pth is not None :
    state_dict = torch.load(config.model_pth)
    model.load_state_dict(state_dict, strict=True)
    print_and_log_info(logger, "Loading model path from {} ".format(config.model_pth))
    
def write(metric_name, metric_val, iter,expr_dir):
    acc_log_dir = os.path.join(expr_dir, 'acc_log.txt')
    
    # Add seed information when writing for the first time
    if not os.path.exists(acc_log_dir):
        with open(acc_log_dir, 'w') as f:
            f.write(f'Seed: {args.seed}\n')
    
    # Append
    with open(acc_log_dir, 'a') as llog:
        llog.write(f' {iter + 1}\n')
        line = f'{metric_name}: {np.mean(metric_val)} ' 
        line += ' '.join([f'{x}' for x in metric_val]) + '\n'
        llog.write(line)
        
##### ------ training ------- #####
nb_iter = 0
avg_loss = 0.
avg_lr = 0.
min_mpjpe_mocap=100000
while (nb_iter + 1) < config.cos_lr_total_iters:
    print(f"{nb_iter + 1} / {config.cos_lr_total_iters}")
    
    if config.dataset == 'h36m':
        pass
    else:# Create data generator, config.num_train_samples is the number of samples in an epoch
        train_generator = dataset.sampling_generator(num_samples=config.num_train_samples, batch_size=config.batch_size)
        data_source = train_generator

    for (h36m_motion_input, h36m_motion_target) in data_source:
        # B,P,T,JK
        h36m_motion_input=torch.tensor(h36m_motion_input).float()
        h36m_motion_target=torch.tensor(h36m_motion_target).float()
        if nb_iter==10:
            print(h36m_motion_input.sum(),'****************')
        loss, optimizer, current_lr = train_step(h36m_motion_input, h36m_motion_target, model, optimizer, nb_iter, config.cos_lr_total_iters, config.cos_lr_max, config.cos_lr_min)
        
        if nb_iter == 99:# For debugging
            print("第100个iter的loss:",loss)#第100个iter的loss: 0.35467132925987244
            
        avg_loss += loss
        avg_lr += current_lr
        # Print loss
        if (nb_iter + 1) % config.print_every ==  0 :
            avg_loss = avg_loss / config.print_every
            avg_lr = avg_lr / config.print_every

            print_and_log_info(logger, "Iter {} Summary: ".format(nb_iter + 1))
            print_and_log_info(logger, f"\t lr: {avg_lr} \t Training loss: {avg_loss}")
            avg_loss = 0
            avg_lr = 0
        # Save model and evaluate model
        if (nb_iter + 1) % config.save_every ==  0 or nb_iter==0:
            with torch.no_grad():
                # torch.save(model.state_dict(), config.snapshot_dir + '/model-iter-' + str(nb_iter + 1) + '.pth')
                model.eval()
                if config.dataset=="h36m":
                    pass
                else:
                    print("begin test")
                    
                    eval_generator_mocap = eval_dataset_mocap.iter_generator(batch_size=config.batch_size)#按顺序遍历一次数据集
                    mpjpe_res_mocap,vim_res_mocap,jpe_res_mocap,ape_res_mocap,fde_res_mocap=mpjpe_vim_test(config, model, eval_generator_mocap,is_mocap=True,select_vim_frames=[1, 5, 10, 20, 29],select_mpjpe_frames=[10,20,30])#得到评估结果
                    
                    # if mpjpe_res_mocap[0]<min_mpjpe_mocap:#保存最好的模型
                    #     min_mpjpe_mocap=mpjpe_res_mocap[0]
                    #     torch.save(model.state_dict(), config.snapshot_dir + '/model-iter-' + str(nb_iter + 1) + '.pth')
                    #     print("save model")
                    eval_generator_mupots = eval_dataset_mupots.iter_generator(batch_size=config.batch_size)
                    mpjpe_res_mupots,vim_res_mupots,jpe_res_mupots,ape_res_mupots,fde_res_mupots=mpjpe_vim_test(config, model, eval_generator_mupots,is_mocap=False,select_vim_frames=[1, 5, 10, 20, 29],select_mpjpe_frames=[10,20,30])
                    
                    # print(f"iter:{nb_iter},mpjpe_mocap:",mpjpe_res_mocap)#50->10:[0.09955118384316544];40->20:[0.16432605807057066]
                    # print(f"iter:{nb_iter},mpjpe_mupots:",mpjpe_res_mupots)#50->10:[0.09054746448865943];40->20:[0.15342454510210662]
                    # print(f"iter:{nb_iter},vim_mocap:",vim_res_mocap)#50->10:[0.09054746448865943];40->20:[0.15342454510210662]
                    # print(f"iter:{nb_iter},vim_mupots:",vim_res_mupots)#50->10:[0.09054746448865943];40->20:[0.15342454510210662]
                    
                    write('mpjpe_mocap',mpjpe_res_mocap,nb_iter,config.expr_dir)
                    write('vim_mocap',vim_res_mocap,nb_iter,config.expr_dir)
                    write('jpe_mocap',jpe_res_mocap,nb_iter,config.expr_dir)
                    write('ape_mocap',ape_res_mocap,nb_iter,config.expr_dir)
                    write('fde_mocap',fde_res_mocap,nb_iter,config.expr_dir)
                    
                    write('mpjpe_mupots',mpjpe_res_mupots,nb_iter,config.expr_dir)
                    write('vim_mupots',vim_res_mupots,nb_iter,config.expr_dir)
                    write('jpe_mupots',jpe_res_mupots,nb_iter,config.expr_dir)
                    write('ape_mupots',ape_res_mupots,nb_iter,config.expr_dir)
                    write('fde_mupots',fde_res_mupots,nb_iter,config.expr_dir)
                    
                model.train()
        # Visualize model
        if ((nb_iter + 1) % config.vis_every ==  0  ) and config.dataset!="h36m":
            model.eval()
            with torch.no_grad():
                if True:
                    h36m_motion_input = eval_dataset_mocap.sample()[:,:,:config.t_his]
                    h36m_motion_input=torch.tensor(h36m_motion_input,device=config.device).float()
                    h36m_motion_input=h36m_motion_input[:1]#1，p,t,jk
                    motion_pred=regress_pred(model,h36m_motion_input,config)# Predicted pose
                    
                    b,p,n,c = motion_pred.shape
                    motion_pred = motion_pred.reshape(b,p,n,config.n_joint,3)
                    h36m_motion_input=h36m_motion_input.reshape(b,p,config.t_his,config.n_joint,3)
                    motion=torch.cat([h36m_motion_input,motion_pred],dim=2).cpu().detach().numpy()
                    visuaulize(motion,f"iter:{nb_iter}",config.vis_dir,input_len=30)# Visualize
                else:# Only show gt
                    h36m_motion = eval_dataset_mocap.sample()
                    h36m_motion=torch.tensor(h36m_motion,device=config.device).float()
                    h36m_motion=h36m_motion[:1]#1，p,t,jk
                    
                    b,p,n,c = h36m_motion.shape
                    h36m_motion=h36m_motion.reshape(b,p,n,config.n_joint,3)
                    motion=h36m_motion.cpu().detach().numpy()
                    visuaulize(motion,f"iter:{nb_iter}",config.vis_dir)# Visualize
            model.train()
        if (nb_iter + 1) == config.cos_lr_total_iters :
            break
        nb_iter += 1

writer.close()
