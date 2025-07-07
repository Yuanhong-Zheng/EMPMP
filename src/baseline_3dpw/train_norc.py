import argparse
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
print('当前工作目录：',os.getcwd())
import json
from src.models_dual_inter_traj_3dpw.utils import Get_RC_Data,visuaulize,visuaulize_bianhao,seed_set,get_dct_matrix,gen_velocity,predict,update_metric,getRandomPermuteOrder,getRandomRotatePoseTransform,paixu_person
from lr import update_lr_multistep
from src.baseline_3dpw.config import config
from src.models_dual_inter_traj_3dpw.model import siMLPe as Model
from src.baseline_3dpw.lib.dataset.dataset_3dpw import get_3dpw_dataloader
from src.baseline_3dpw.lib.utils.logger import get_logger, print_and_log_info
from src.baseline_3dpw.lib.utils.pyt_utils import  ensure_dir
import torch
from torch.utils.tensorboard import SummaryWriter
from src.baseline_3dpw.test import vim_test,random_pred,mpjpe_vim_test
import shutil
from easydict import EasyDict as edict
import warnings
from src.models_dual_inter_traj_3dpw.model import inverse_sort_tensor
# Ignore all warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--exp-name', type=str, default="+(4)", help='=exp name')
parser.add_argument('--dataset', type=str, default="others", help='=exp name')
parser.add_argument('--seed', type=int, default=800, help='=seed')
parser.add_argument('--temporal-only', action='store_true', help='=temporal only')
parser.add_argument('--layer-norm-axis', type=str, default='spatial', help='=layernorm axis')
parser.add_argument('--with-normalization', type=bool,default=True, help='=use layernorm')#无用
parser.add_argument('--spatial-fc', action='store_true', help='=use only spatial fc')
parser.add_argument('--normalization',type=bool,default=True, help='Normalize data')
parser.add_argument('--norm_way',type=str,default='first', help='=use only spatial fc')
parser.add_argument('--rc',type=bool,default=False, help='=use only spatial fc')
parser.add_argument('--permute_p', type=bool, default=True, help='Permute P dimension')
parser.add_argument('--random_rotate', type=bool, default=True, help='Random rotation around world center')
parser.add_argument('--num', type=int, default=64, help='=num of blocks')
parser.add_argument('--paixu', type=bool, default=False, help='=num of blocks')
parser.add_argument('--global_num', type=int, default=4, help='=num of global blocks')
parser.add_argument('--interaction_interval', type=int, default=4, help='Interval between local and Global interactions, must be divisible by num')
parser.add_argument('--weight', type=float, default=1., help='=loss weight')
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--n_p', type=int, default=2)
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--vis_every', type=int, default=250000000000)
parser.add_argument('--save_every', type=int, default=100)
parser.add_argument('--print_every', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
args = parser.parse_args()

# Create folder
expr_dir = os.path.join('exprs', args.exp_name)
if os.path.exists(expr_dir):
    shutil.rmtree(expr_dir)
os.makedirs(expr_dir, exist_ok=True)

# Ensure reproducibility
seed_set(args.seed)
torch.use_deterministic_algorithms(True)

# Record metric file
acc_log_dir = os.path.join(expr_dir, 'acc_log.txt')
acc_log = open(acc_log_dir, 'a')
acc_log.write(''.join('Seed : ' + str(args.seed) + '\n'))
acc_log.flush()

acc_best_log_dir = os.path.join(expr_dir, 'acc_best_log.txt')
acc_best_log=open(acc_best_log_dir, 'a')
acc_best_log.write(''.join('Seed : ' + str(args.seed) + '\n'))
acc_best_log.flush()

# Configuration
config.rc=args.rc
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
config.motion_mlp.num_global_layers = args.global_num
config.motion_mlp.n_p=args.n_p
config.motion_mlp.interaction_interval = args.interaction_interval
config.snapshot_dir=os.path.join(expr_dir, 'snapshot')
ensure_dir(config.snapshot_dir)#创建文件夹
config.vis_dir=os.path.join(expr_dir, 'vis')
ensure_dir(config.vis_dir)#创建文件夹
config.log_file=os.path.join(expr_dir, 'log.txt')
config.model_pth=args.model_path
config.paixu=args.paixu
#
writer = SummaryWriter()

# Get DCT matrix
dct_m,idct_m = get_dct_matrix(config.dct_len)
dct_m = torch.tensor(dct_m).float().to(config.device).unsqueeze(0)
idct_m = torch.tensor(idct_m).float().to(config.device).unsqueeze(0)
config.dct_m=dct_m
config.idct_m=idct_m

def train_step(h36m_motion_input, h36m_motion_target,padding_mask, model, optimizer, nb_iter, total_iter, max_lr, min_lr) :
    if args.random_rotate:
        h36m_motion_input,h36m_motion_target=getRandomRotatePoseTransform(h36m_motion_input,h36m_motion_target)
    if args.paixu:#DCT之前
        h36m_motion_input,h36m_motion_target,sorted_indices=paixu_person(h36m_motion_input,h36m_motion_target)
    if args.permute_p:
        h36m_motion_input,h36m_motion_target=getRandomPermuteOrder(h36m_motion_input,h36m_motion_target)
    if config.rc:
        h36m_motion_input,h36m_motion_target=Get_RC_Data(h36m_motion_input,h36m_motion_target)
    # for_vis=h36m_motion_target[:1].reshape(1,2,14,-1,3).cpu().detach().numpy()
    # visuaulize(for_vis,'正确输出，rc前','可视化测试',input_len=16,dataset='3dpw')    
    motion_pred=predict(model,h36m_motion_input,config,h36m_motion_target)#b,p,n,c
    b,p,n,c = h36m_motion_target.shape
    if args.paixu:#DCT之后
        motion_pred=inverse_sort_tensor(motion_pred,sorted_indices)#b,p,n,c
        h36m_motion_target=inverse_sort_tensor(h36m_motion_target,sorted_indices)
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
        # loss = loss#去掉速度损失
    else:
        #todo:加上mask
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

dataloader_train=get_3dpw_dataloader(split="train",cfg=config,shuffle=True)#-1040.5767
dataloader_test=get_3dpw_dataloader(split="jrt",cfg=config,shuffle=True)#-5.9048
dataloader_test_sample=get_3dpw_dataloader(split="jrt",cfg=config,shuffle=True,batch_size=1)#-5.9048
random_iter=iter(dataloader_test_sample)

# initialize optimizer
optimizer = torch.optim.Adam(model.parameters(),
                             lr=config.cos_lr_max,
                             weight_decay=config.weight_decay)
# Create logger
def default_serializer(obj):
    if isinstance(obj, torch.Tensor):
        return None  # 或者返回一个标记，或将其转换为可序列化的类型
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

logger = get_logger(config.log_file, 'train')
print_and_log_info(logger, json.dumps(config, indent=4, sort_keys=True,default=default_serializer))

if config.model_pth is not None :#用于预训练
    state_dict = torch.load(config.model_pth,map_location=config.device)
    model.load_state_dict(state_dict, strict=True)
    print_and_log_info(logger, "Loading model path from {} ".format(config.model_pth))
    print("Loading model path from {} ".format(config.model_pth))
##### ------ Training ------- #####
nb_iter = 0
avg_loss = 0.
avg_lr = 0.
min_vim=100000
metric_best=edict()

def write(metric_name,metric_val,iter,llog):
    llog.write(''.join(str(iter + 1) + '\n'))
    
    line = f'{metric_name}:'         
    line+=str(metric_val.mean())+' '
    for ii in metric_val:
        line += str(ii) + ' '
    line += '\n'
    llog.write(''.join(line))

    llog.flush()

while (nb_iter + 1) < config.cos_lr_total_iters:
    print(f"{nb_iter + 1} / {config.cos_lr_total_iters}")
    # joint_list=[]
    # for (joints, masks, padding_mask) in dataloader_train:
    #     h36m_motion_input=joints[:,:,:config.t_his].flatten(-2)#16
    #     h36m_motion_target=joints[:,:,config.t_his:].flatten(-2)#14
        
    #     h36m_motion_input=torch.tensor(h36m_motion_input).float()
    #     h36m_motion_target=torch.tensor(h36m_motion_target).float()
        
    #     h36m_motion_input,h36m_motion_target=Get_RC_Data(h36m_motion_input,h36m_motion_target)
    #     joints=torch.cat([h36m_motion_input,h36m_motion_target],dim=2).reshape(-1,2,30,13,3)
    #     joint_list.append(joints)
    # import pdb;pdb.set_trace()
    # joint_list=torch.cat(joint_list,dim=0)
    # torch.save(joint_list,'joint_list.pt',_use_new_zipfile_serialization=False)
    
    for (joints, masks, padding_mask) in dataloader_train:
        
        # B,P,T,JK
        h36m_motion_input=joints[:,:,:config.t_his].flatten(-2)#16
        h36m_motion_target=joints[:,:,config.t_his:].flatten(-2)#14
        
        h36m_motion_input=torch.tensor(h36m_motion_input).float()
        h36m_motion_target=torch.tensor(h36m_motion_target).float()
        
        loss, optimizer, current_lr = train_step(h36m_motion_input, h36m_motion_target,padding_mask, model, optimizer, nb_iter, config.cos_lr_total_iters, config.cos_lr_max, config.cos_lr_min)
        
        if nb_iter == 9:
            print("第10个iter的loss:",loss)#0.36073487997055054
            
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
                
                print("begin test")
                
                # vim_3dpw=vim_test(config, model, dataloader_test,dataset="3dpw")
                mpjpe,vim,jpe,ape,fde=mpjpe_vim_test(config, model, dataloader_test,is_mocap=False,select_vim_frames=[1, 3, 7, 9, 13],select_mpjpe_frames=[7,14,14])
                
                print(f"iter:{nb_iter},vim:",vim)
                print(f"iter:{nb_iter},mpjpe:",mpjpe)
                
                update_metric(metric_best,"vim",vim,nb_iter)
                update_metric(metric_best,"mpjpe",mpjpe,nb_iter)
                update_metric(metric_best,"jpe",jpe,nb_iter)
                update_metric(metric_best,"ape",ape,nb_iter)
                update_metric(metric_best,"fde",fde,nb_iter)
                ##log acc_log
                
                # if min_vim>vim.mean():
                #     min_vim=vim.mean()
                #     torch.save(model.state_dict(), config.snapshot_dir + '/model-best' + '.pth')
                # torch.save(model.state_dict(), config.snapshot_dir + '/model-iter-' + str(nb_iter + 1) + '.pth')
                
                # line = 'vim_3dpw:'         
                # line+=str(vim.mean())+' '
                # for ii in vim:
                #     line += str(ii) + ' '
                # line += '\n'
                # acc_log.write(''.join(line))

                # acc_log.flush()
                write("vim",vim,nb_iter,acc_log)
                write("mpjpe",mpjpe,nb_iter,acc_log)
                write("jpe",jpe,nb_iter,acc_log)
                write("ape",ape,nb_iter,acc_log)
                write("fde",fde,nb_iter,acc_log)

                write("vim",metric_best.vim.val,metric_best.vim.iter,acc_best_log)
                write("mpjpe",metric_best.mpjpe.val,metric_best.mpjpe.iter,acc_best_log)
                write("jpe",metric_best.jpe.val,metric_best.jpe.iter,acc_best_log)
                write("ape",metric_best.ape.val,metric_best.ape.iter,acc_best_log)
                write("fde",metric_best.fde.val,metric_best.fde.iter,acc_best_log)
                
                model.train()
        # Visualize model
        if ((nb_iter + 1) % config.vis_every ==  0 ) :
            model.eval()
            with torch.no_grad():  
                
                h36m_motion_input,motion_pred=random_pred(config=config,model=model,iter=random_iter)
                
                if h36m_motion_input is not None:
                    b,p,n,c = motion_pred.shape
                    motion_pred = motion_pred.reshape(b,p,n,config.n_joint,3)
                    h36m_motion_input=h36m_motion_input.reshape(b,p,config.t_his,config.n_joint,3)
                    motion=torch.cat([h36m_motion_input,motion_pred],dim=2).cpu().detach().numpy()
                    visuaulize(motion,f"iter:{nb_iter}",config.vis_dir,input_len=15,dataset='mupots')
                
                model.train()
        if (nb_iter + 1) == config.cos_lr_total_iters :
            break
        nb_iter += 1

writer.close()
