import argparse
import os, sys
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import json
import numpy as np
from utils import visuaulize,seed_set,get_dct_matrix,update_lr_multistep,gen_velocity
from src.baseline_h36m_40to20.config import config
from src.baseline_h36m_40to20.model import siMLPe as Model
from src.baseline_h36m_40to20.lib.datasets.dataset_mocap import DATA
from lib.utils.logger import get_logger, print_and_log_info
from lib.utils.pyt_utils import  ensure_dir
import torch
from torch.utils.tensorboard import SummaryWriter
from src.baseline_h36m_40to20.test import mpjpe_test_regress,regress_pred
import shutil

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--exp-name', type=str, default="复现：40->20,长时间观察", help='=exp name')
parser.add_argument('--dataset', type=str, default="others", help='=exp name')
parser.add_argument('--seed', type=int, default=888, help='=seed')
parser.add_argument('--temporal-only', action='store_true', help='=temporal only')
parser.add_argument('--layer-norm-axis', type=str, default='spatial', help='=layernorm axis')
parser.add_argument('--with-normalization', type=bool,default=True, help='=use layernorm')
parser.add_argument('--spatial-fc', action='store_true', help='=use only spatial fc')
parser.add_argument('--num', type=int, default=64, help='=num of blocks')
parser.add_argument('--weight', type=float, default=1., help='=loss weight')
parser.add_argument('--device', type=str, default="cuda:2")
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--n_p', type=int, default=3)
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--vis_every', type=int, default=1000)
parser.add_argument('--save_every', type=int, default=1000)
parser.add_argument('--print_every', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=256)
args = parser.parse_args()

# 创建文件夹
expr_dir = os.path.join('exprs', args.exp_name)
if os.path.exists(expr_dir):
    shutil.rmtree(expr_dir)
os.makedirs(expr_dir, exist_ok=True)

#确保实验课重现
seed_set(args.seed)
torch.use_deterministic_algorithms(True)

#记录指标的文件
acc_log_dir = os.path.join(expr_dir, 'acc_log.txt')
acc_log = open(acc_log_dir, 'a')
acc_log.write(''.join('Seed : ' + str(args.seed) + '\n'))
acc_log.flush()

#配置
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
config.snapshot_dir=os.path.join(expr_dir, 'snapshot')
ensure_dir(config.snapshot_dir)#创建文件夹
config.vis_dir=os.path.join(expr_dir, 'vis')
ensure_dir(config.vis_dir)#创建文件夹
config.log_file=os.path.join(expr_dir, 'log.txt')
config.model_pth=args.model_path

#
writer = SummaryWriter()

#获取dct矩阵
dct_m,idct_m = get_dct_matrix(config.dct_len)
dct_m = torch.tensor(dct_m).float().to(config.device).unsqueeze(0)
idct_m = torch.tensor(idct_m).float().to(config.device).unsqueeze(0)
config.dct_m=dct_m
config.idct_m=idct_m

def train_step(h36m_motion_input, h36m_motion_target, model, optimizer, nb_iter, total_iter, max_lr, min_lr) :

    if config.deriv_input:
        b,p,n,c = h36m_motion_input.shape
        h36m_motion_input_ = h36m_motion_input.clone()
        #b,p,n,c
        h36m_motion_input_ = torch.matmul(dct_m[:, :, :config.dct_len], h36m_motion_input_.to(config.device))
    else:
        h36m_motion_input_ = h36m_motion_input.clone()

    motion_pred = model(h36m_motion_input_.to(config.device))
    motion_pred = torch.matmul(idct_m[:, :config.dct_len, :], motion_pred)#b,p,n,c

    if config.deriv_output:
        offset = h36m_motion_input[:, :,-1:].to(config.device)#b,p,1,c
        motion_pred = motion_pred[:,:, :config.t_pred] + offset#b,p,n,c
    else:
        motion_pred = motion_pred[:, :config.t_pred]

    b,p,n,c = h36m_motion_target.shape
    #预测的姿态
    motion_pred = motion_pred.reshape(b,p,n,config.n_joint,3).reshape(-1,3)
    #GT
    h36m_motion_target = h36m_motion_target.to(config.device).reshape(b,p,n,config.n_joint,3).reshape(-1,3)
    #计算loss
    loss = torch.mean(torch.norm(motion_pred - h36m_motion_target, 2, 1))

    if config.use_relative_loss:
        motion_pred = motion_pred.reshape(b,p,n,config.n_joint,3)
        dmotion_pred = gen_velocity(motion_pred)#计算速度
        
        motion_gt = h36m_motion_target.reshape(b,p,n,config.n_joint,3)
        dmotion_gt = gen_velocity(motion_gt)#计算速度
        
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

#创建模型
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
#创建logger
def default_serializer(obj):
    if isinstance(obj, torch.Tensor):
        return None  # 或者返回一个标记，或将其转换为可序列化的类型
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

logger = get_logger(config.log_file, 'train')
print_and_log_info(logger, json.dumps(config, indent=4, sort_keys=True,default=default_serializer))

if config.model_pth is not None :
    state_dict = torch.load(config.model_pth)
    model.load_state_dict(state_dict, strict=True)
    print_and_log_info(logger, "Loading model path from {} ".format(config.model_pth))

##### ------ training ------- #####
nb_iter = 0
avg_loss = 0.
avg_lr = 0.

while (nb_iter + 1) < config.cos_lr_total_iters:
    print(f"{nb_iter + 1} / {config.cos_lr_total_iters}")
    
    if config.dataset == 'h36m':
        pass
    else:
        train_generator = dataset.sampling_generator(num_samples=config.num_train_samples, batch_size=config.batch_size)
        data_source = train_generator

    for (h36m_motion_input, h36m_motion_target) in data_source:
        # B,P,T,JK
        h36m_motion_input=torch.tensor(h36m_motion_input).float()
        h36m_motion_target=torch.tensor(h36m_motion_target).float()
        
        loss, optimizer, current_lr = train_step(h36m_motion_input, h36m_motion_target, model, optimizer, nb_iter, config.cos_lr_total_iters, config.cos_lr_max, config.cos_lr_min)
        
        if nb_iter == 0:
            print("第一个iter的loss:",loss)#50->10:0.18656916916370392;40->10:0.1986370086669922
            
        avg_loss += loss
        avg_lr += current_lr
        #打印损失
        if (nb_iter + 1) % config.print_every ==  0 :
            avg_loss = avg_loss / config.print_every
            avg_lr = avg_lr / config.print_every

            print_and_log_info(logger, "Iter {} Summary: ".format(nb_iter + 1))
            print_and_log_info(logger, f"\t lr: {avg_lr} \t Training loss: {avg_loss}")
            avg_loss = 0
            avg_lr = 0
        #保存模型并评估模型
        if (nb_iter + 1) % config.save_every ==  0 or nb_iter==0:
            with torch.no_grad():
                torch.save(model.state_dict(), config.snapshot_dir + '/model-iter-' + str(nb_iter + 1) + '.pth')
                model.eval()
                if config.dataset=="h36m":
                    pass
                else:
                    print("begin test")
                    
                    eval_generator_mocap = eval_dataset_mocap.iter_generator(batch_size=1)
                    mpjpe_res_mocap=mpjpe_test_regress(config, model, eval_generator_mocap,dataset="mocap")
                    
                    eval_generator_mupots = eval_dataset_mupots.iter_generator(batch_size=1)
                    mpjpe_res_mupots=mpjpe_test_regress(config, model, eval_generator_mupots,dataset="mupots")
                    
                    print(f"iter:{nb_iter},mpjpe_mocap:",mpjpe_res_mocap)#50->10:[0.09955118384316544];40->20:[0.16432605807057066]
                    print(f"iter:{nb_iter},mpjpe_mocap:",mpjpe_res_mupots)#50->10:[0.09054746448865943];40->20:[0.15342454510210662]

                    acc_log.write(''.join(str(nb_iter + 1) + '\n'))
                    
                    line = 'mpjpe_mocap:'
                    for ii in mpjpe_res_mocap:
                        line += str(ii) + ' '
                    line += '\n'
                    acc_log.write(''.join(line))
                    
                    line='mpjpe_mupots:'
                    for ii in mpjpe_res_mupots:
                        line += str(ii) + ' '
                    line += '\n'
                    acc_log.write(''.join(line))
                    
                    acc_log.flush()
                model.train()
        #可视化模型
        if ((nb_iter + 1) % config.vis_every ==  0 or nb_iter==0) and config.dataset!="h36m":
            model.eval()
            with torch.no_grad():  
                h36m_motion_input = eval_dataset_mocap.sample()[:,:,:config.t_his]
                h36m_motion_input=torch.tensor(h36m_motion_input,device=config.device).float()
                h36m_motion_input=h36m_motion_input[:1]#1，p,t,jk
                motion_pred=regress_pred(model,h36m_motion_input,config)
                
                b,p,n,c = motion_pred.shape
                motion_pred = motion_pred.reshape(b,p,n,config.n_joint,3)
                h36m_motion_input=h36m_motion_input.reshape(b,p,config.t_his,config.n_joint,3)
                motion=torch.cat([h36m_motion_input,motion_pred],dim=2).cpu().detach().numpy()
                visuaulize(motion,f"iter:{nb_iter}",config.vis_dir)
                
                model.train()
        if (nb_iter + 1) == config.cos_lr_total_iters :
            break
        nb_iter += 1

writer.close()
