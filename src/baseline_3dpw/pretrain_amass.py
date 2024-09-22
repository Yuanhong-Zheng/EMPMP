import argparse
import warnings
# 忽略所有警告
warnings.filterwarnings("ignore")
import os, sys
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from utils import AverageMeter
import json
import random
import numpy as np
from utils import visuaulize,seed_set,get_dct_matrix,gen_velocity,update_metric,update_lr_multistep_mine
from config import config
from model import siMLPe as Model
from lib.utils.logger import get_logger, print_and_log_info
from lib.utils.pyt_utils import  ensure_dir
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.baseline_3dpw.test import vim_test_pretrain,random_pred_pretrain
import shutil
from easydict import EasyDict as edict
import sys
sys.path.append(".")
from lib.dataset.dataset_amass import AMASSDatasets
from torch.utils.data import DataLoader
from lib.utils.config_3dpw import *
from lib.utils.util import rotate_Y

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--exp-name', type=str, default="预训练：3dpw数据集", help='=exp name')
parser.add_argument('--dataset', type=str, default="others", help='=exp name')
parser.add_argument('--seed', type=int, default=888, help='=seed')
parser.add_argument('--temporal-only', action='store_true', help='=temporal only')
parser.add_argument('--layer-norm-axis', type=str, default='spatial', help='=layernorm axis')
parser.add_argument('--with-normalization', type=bool,default=True, help='=use layernorm')
parser.add_argument('--spatial-fc', action='store_true', help='=use only spatial fc')
parser.add_argument('--num', type=int, default=64, help='=num of blocks')
parser.add_argument('--weight', type=float, default=1., help='=loss weight')
parser.add_argument('--device', type=str, default="cuda:3")
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--n_p', type=int, default=3)
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--vis_every', type=int, default=5)
parser.add_argument('--save_every', type=int, default=1)
parser.add_argument('--print_every', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=512)
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

acc_best_log_dir = os.path.join(expr_dir, 'acc_best_log.txt')
acc_best_log=open(acc_best_log_dir, 'a')
acc_best_log.write(''.join('Seed : ' + str(args.seed) + '\n'))
acc_best_log.flush()

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



def train_step(h36m_motion_input, h36m_motion_target,padding_mask, model, optimizer, nb_iter, total_iter, max_lr, min_lr) :

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
    #mask:b,p
    expanded_mask = padding_mask.unsqueeze(-1).repeat(1, 1, n * config.n_joint).reshape(-1)
    #计算loss
    loss = torch.mean(torch.norm(motion_pred - h36m_motion_target, dim=1)[expanded_mask])
    
    if config.use_relative_loss:
        motion_pred = motion_pred.reshape(b,p,n,config.n_joint,3)
        dmotion_pred = gen_velocity(motion_pred)#计算速度
        
        motion_gt = h36m_motion_target.reshape(b,p,n,config.n_joint,3)
        dmotion_gt = gen_velocity(motion_gt)#计算速度
        
        # dloss = torch.mean(torch.norm((dmotion_pred - dmotion_gt).reshape(-1,3), 2, 1))
        expanded_mask = padding_mask.unsqueeze(-1).repeat(1, 1, (n-1)*config.n_joint).reshape(-1)
        dloss=torch.mean(torch.norm((dmotion_pred - dmotion_gt).reshape(-1,3), dim=1)[expanded_mask])
        loss = loss + dloss
    else:
        #todo:加上mask
        loss = loss.mean()

    writer.add_scalar('Loss/angle', loss.detach().cpu().numpy(), nb_iter)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer, current_lr = update_lr_multistep_mine(nb_iter, total_iter, max_lr, min_lr, optimizer)
    writer.add_scalar('LR/train', current_lr, nb_iter)

    return loss.item(), optimizer, current_lr

#创建模型
model = Model(config).to(device=config.device)
model.train()
print(">>> total params: {:.2f}M".format(
    sum(p.numel() for p in list(model.parameters())) / 1000000.0))

amass_path = AMASS_dir
dset_train = AMASSDatasets(path_to_data=amass_path, skel_path=skel_path, input_n=config.t_his, output_n=config.t_pred, split=0,device=config.device)
train_len = len(dset_train)
print("Train set length:", train_len)
train_loader = DataLoader(dset_train, batch_size=config.batch_size, num_workers=4, shuffle=True)
print("Load Train set!")
dset_val =  AMASSDatasets(path_to_data=amass_path, skel_path=skel_path, input_n=config.t_his, output_n=config.t_pred, split=1,device=config.device)
val_len = len(dset_val)
print("Valid set length:", val_len)
valid_loader = DataLoader(dset_val, batch_size=2, num_workers=4, shuffle=True)
iter=iter(valid_loader)
print("Load Valid set!")
dset_test =  AMASSDatasets(path_to_data=amass_path, skel_path=skel_path, input_n=config.t_his, output_n=config.t_pred, split=2,device=config.device)
test_len = len(dset_test)
print("Test set length:", test_len)
test_loader = DataLoader(dset_test, batch_size=config.batch_size, num_workers=4, shuffle=False)
print("Load Test set!")

joint_to_use = np.array([1, 2, 4, 5, 7, 8, 15, 16, 17, 18, 19, 20, 21])

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
nb_iter = 0#这里的iter实际上是epoch

metric_best=edict()

while (nb_iter + 1) < config.epoch:
    print(f"epoch:{nb_iter + 1} / {config.epoch}")
    avg_loss = AverageMeter()
    avg_lr=AverageMeter()
    for i, data in enumerate(train_loader):
        # print(i)
        data = data.float().to(config.device)
        batch_size = data.shape[0]
        if batch_size % 2 != 0:
            continue
        data_to_use = data[:, :, joint_to_use].contiguous().view(batch_size//2, 2, 30, 13, 3)#得到使用的数据,需要旋转:B,p,T,J,K
        
        batch_size = data_to_use.shape[0]
        input_total = data_to_use.permute(0, 1, 3, 2, 4).contiguous()#B,P,J,T,K
        padding_mask=torch.ones(batch_size,2).float().to(config.device).bool()
        
        # B/2, N, J, T, 3
        angle = random.random()*360
        # random rotation
        input_total = rotate_Y(input_total, angle)
        input_total *= (random.random()*0.4+0.8)
        
        input_total[:,:,:,:,[1,2]]=input_total[:,:,:,:,[2,1]]#B,P,J,T,K
        input_total=input_total.transpose(2,3)#B,P,T,J,K
        
        # # vis for one sample
        # for_vis=input_total.clone()[:1].cpu().detach().numpy()
        # visuaulize(for_vis,f"测试：6",'垃圾桶/vis')
        
        # B,P,T,JK
        h36m_motion_input=input_total[:,:,:config.t_his].flatten(-2)#16
        h36m_motion_target=input_total[:,:,config.t_his:].flatten(-2)#14
        
        h36m_motion_input=torch.tensor(h36m_motion_input).float()
        h36m_motion_target=torch.tensor(h36m_motion_target).float()
        
        loss, optimizer, current_lr = train_step(h36m_motion_input, h36m_motion_target,padding_mask, model, optimizer, nb_iter, config.cos_lr_total_iters, config.cos_lr_max, config.cos_lr_min)
        
        if i == 99 and nb_iter==0:#i是iter
            print("第100个iter的loss:",loss)#0.2044074833393097
            
        avg_loss.update(loss,1) 
        avg_lr.update(current_lr,1)
    ##一个epoch结束------------
    #打印损失
    if (nb_iter + 1) % config.print_every ==  0 :

        print_and_log_info(logger, "epoch {} Summary: ".format(nb_iter + 1))
        print_and_log_info(logger, f"\t lr: {avg_lr.avg} \t Training loss: {avg_loss.avg}")
        avg_loss.__init__()
        avg_lr.__init__()
        
    #保存模型并评估模型
    if (nb_iter + 1) % config.save_every ==  0 or nb_iter==0:
        with torch.no_grad():
            torch.save(model.state_dict(), config.snapshot_dir + '/model-iter-' + str(nb_iter + 1) + '.pth')
            model.eval()
            
            print("begin test")
            
            vim_3dpw=vim_test_pretrain(config, model, test_loader,joint_to_use=joint_to_use,dataset="3dpw")
            
            print(f"iter:{nb_iter},vim_3dpw:",vim_3dpw)#[ 29.349432  53.55681   94.51841  112.672714 143.09254 ]
            
            update_metric(metric_best,"vim",vim_3dpw,nb_iter)
            
            ##log acc
            acc_log.write(''.join(str(nb_iter + 1) + '\n'))
            
            line = 'vim_3dpw:'
            line+=str(vim_3dpw.mean())+' '
            for ii in vim_3dpw:
                line += str(ii) + ' '
            line += '\n'
            acc_log.write(''.join(line))

            acc_log.flush()
            
            ##log best acc
            acc_best_log.write(''.join(str(metric_best.vim.iter + 1) + '\n'))
            
            line = 'vim_3dpw:'
            line+=str(metric_best.vim.avg)+' '
            for ii in metric_best.vim.val:
                line += str(ii) + ' '
            line += '\n'
            acc_best_log.write(''.join(line))

            acc_best_log.flush()
            
            ##save model if best
            if metric_best.vim.iter == nb_iter:
                torch.save(model.state_dict(), config.snapshot_dir + '/model-best.pth')
            
            model.train()
            
    #可视化模型
    if ((nb_iter + 1) % config.vis_every ==  0 or nb_iter==0) :
        model.eval()
        with torch.no_grad():  
            
            h36m_motion_input,motion_pred=random_pred_pretrain(config=config,model=model,iter=iter,joint_to_use=joint_to_use)
            
            if h36m_motion_input is not None:
                b,p,n,c = motion_pred.shape
                motion_pred = motion_pred.reshape(b,p,n,config.n_joint,3)
                h36m_motion_input=h36m_motion_input.reshape(b,p,config.t_his,config.n_joint,3)
                motion=torch.cat([h36m_motion_input,motion_pred],dim=2).cpu().detach().numpy()
                visuaulize(motion,f"iter:{nb_iter}",config.vis_dir)
            
            model.train()
            
    nb_iter += 1

writer.close()
