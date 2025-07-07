import os
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
from io import BytesIO
import random
import torch
from easydict import EasyDict as edict
from torchvision import transforms

def Get_RC_Data(motion_input,motion_target):#B,P,T,JK. Create 3dpw/rc dataset, mocap/mupots dataset not needed
    #加上速度维度
    b,p,t_in,jk=motion_input.shape
    _,_,t_out,_=motion_target.shape
    t=t_in+t_out
    k=3
    j=jk//k
    motion=torch.cat((motion_input,motion_target),dim=2).reshape(b,p,t,j,k)#B,P,T_in+T_out,J,K
    
    vel_data = torch.zeros((b, p, t, j,k)).to(motion.device)#b,p,T,J,3
    vel_data[:,:,1:,:,:] = (torch.roll(motion, shifts=-1, dims=2) - motion)[:,:,:-1,:,:]#roll是在时间轴上向左滚动一帧，vel_data的第零帧为0
    data = torch.cat((motion, vel_data), dim=-1)#B,P,T,J,6
    data=data.transpose(1,2)#B,T,P,J,6
            
    camera_vel = data[:, 1:t, :, :, 3:].mean(dim=(1, 2, 3)) # B, 3
    data[..., 3:] -= camera_vel[:, None, None, None]
    data[..., :3] = data[:, 0:1, :, :, :3] + data[..., 3:].cumsum(dim=1)
    
    data=data.transpose(1,2)[...,:3].reshape(b,p,t,jk)#B,P,T,jk
    
    return data[:,:,:t_in],data[:,:,t_in:]

def getRandomScaleTransform(joints_input,joints_target,r1=0.8, r2=1.2):
    #scale = (r1 - r2) * torch.rand(1) + r2
    scale = (r1 - r2) * torch.rand(joints_input.shape[0]).reshape(-1, 1, 1, 1) + r2
    return joints_input * scale.to(joints_input.device), joints_target * scale.to(joints_target.device)
def getRandomRotatePoseTransform(joints_input, joints_target):
    """
    Performs a random rotation about the origin (0, 0, 0)
    """

    K=3
    B, P, T_in,JK = joints_input.shape
    _,_,T_out,_=joints_target.shape
    J=JK//K
    
    joints_input=joints_input.reshape(B,P,T_in,J,K)#B,P,T,J,K
    joints_target=joints_target.reshape(B,P,T_out,J,K)#B,P,T,J,K
    
    angles = torch.deg2rad(torch.rand(B)*360)

    rotation_matrix = torch.zeros(B, 3, 3).to(joints_input.device)
    rotation_matrix[:,1,1] = 1
    rotation_matrix[:,0,0] = torch.cos(angles)
    rotation_matrix[:,0,2] = torch.sin(angles)
    rotation_matrix[:,2,0] = -torch.sin(angles)
    rotation_matrix[:,2,2] = torch.cos(angles)

    joints_input = torch.bmm(joints_input.reshape(B, -1, 3).float(), rotation_matrix)
    joints_input = joints_input.reshape(B,P,T_in,JK)
    
    joints_target = torch.bmm(joints_target.reshape(B, -1, 3).float(), rotation_matrix)
    joints_target = joints_target.reshape(B,P,T_out,JK)
    return joints_input,joints_target

def update_metric(metric_dict,metric,value,iter):
    if not hasattr(metric_dict, metric):
        setattr(metric_dict, metric, edict())
    
    metric_dict=getattr(metric_dict, metric)
    
    if not hasattr(metric_dict,'val'):
        setattr(metric_dict, 'val', value)
        setattr(metric_dict, 'iter', iter)
        setattr(metric_dict, 'avg', value.mean())
    else:
        if getattr(metric_dict, 'avg')>value.mean():
            setattr(metric_dict, 'val', value)
            setattr(metric_dict, 'iter', iter)
            setattr(metric_dict, 'avg', value.mean())
            
def getRandomPermuteOrder(joints_input, joints_target):
    """
    Randomly permutes persons across the input token dimension. This helps
    expose all learned embeddings to a variety of poses.
    """
    K=3
    B,N,F_in,JK=joints_input.shape
    _,_,F_out,_=joints_target.shape
    J=JK//K
    perm = torch.argsort(torch.rand(B, N), dim=-1).reshape(B, N)
    idx = torch.arange(B).unsqueeze(-1)
    
    joints_input = joints_input.view(B, N, F_in, J, K)[idx, perm]
    joints_input = joints_input.reshape(B, N, F_in, JK) 
    
    joints_target = joints_target.view(B, N, F_out, J, K)[idx, perm]
    joints_target = joints_target.reshape(B, N, F_out, JK)
    return joints_input,joints_target

class AverageMeter(object):
    """
    From https://github.com/mkocabas/VIBE/blob/master/lib/core/trainer.py
    Keeps track of a moving average.
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def predict(model,h36m_motion_input,config,h36m_motion_target=None):
    b,p,t,jk=h36m_motion_input.shape
    traj=h36m_motion_input.clone().reshape(b,p,t,jk//3,3)#b,p,t,j,3
    if config.normalization:
        h36m_motion_input_,h36m_motion_input,mean=data_normalization(h36m_motion_input,config=None,way=config.norm_way)
        # h36m_motion_input_ = torch.matmul(config.dct_m[:, :config.t_his, :config.t_his], h36m_motion_input_.to(config.device))#归一化后dct
    else:
        h36m_motion_input_ = h36m_motion_input.clone()
        #b,p,n,c
        # h36m_motion_input_ = torch.matmul(config.dct_m[:, :config.t_his, :config.t_his], h36m_motion_input_.to(config.device))
    motion_pred = model(h36m_motion_input_.to(config.device),traj.to(config.device))
    # from thop import profile
    # flops, params = profile(model, inputs=(h36m_motion_input_.to(config.device),traj.to(config.device)))

    # print('FLOPs = ' + str(flops/1000**3) + 'G')

    # print('Params = ' + str(params/1000**2) + 'M')
    # motion_pred = torch.matmul(config.idct_m[:, :config.t_pred, :config.t_pred], motion_pred)#b,p,n,c，idct
    #反归一化
    if config.normalization:
        motion_pred=data_denormalization(motion_pred,h36m_motion_input,mean,config)
    else:
        offset = h36m_motion_input[:, :,-1:].to(config.device)#b,p,1,c
        motion_pred = motion_pred[:,:, :config.t_pred] + offset#b,p,n,c
    return motion_pred

def data_denormalization(motion_pred,h36m_motion_input,mean,config=None):
    b,p,_,c = motion_pred.shape
    offset = h36m_motion_input[:, :,-1:].to(config.device)#b,p,1,c
    motion_pred = motion_pred[:,:, :config.t_pred] + offset#b,p,n,c
    #mean:b,p,1,1,3
    mean=mean.to(config.device)
    motion_pred = motion_pred.reshape(b,p,config.t_pred,-1,3)#b,p,n,j,3
    motion_pred = motion_pred + mean
    motion_pred = motion_pred.reshape(b,p,config.t_pred,-1)#b,p,n,c
    return motion_pred
def data_normalization(h36m_motion_input,config=None,way='all'):
    b,p,n,c = h36m_motion_input.shape
    h36m_motion_input_ = h36m_motion_input.clone()
    
    #归一化
    h36m_motion_input_=h36m_motion_input_.reshape(b,p,n,-1,3)
    #b,p,n,c
    if way=='all':
        mean=h36m_motion_input_[:,:,:1,:,:].mean(dim=3,keepdim=True)#第一帧的平均位置,b,p,1,1,3
    elif way=='first':
        mean=h36m_motion_input_[:,:1,:1,:,:].mean(dim=3,keepdim=True)#第一个人第一帧的平均位置,b,1,1,1,3
    h36m_motion_input_=h36m_motion_input_-mean
    h36m_motion_input_=h36m_motion_input_.reshape(b,p,n,-1)#回到b,p,n,c
    
    h36m_motion_input=h36m_motion_input.reshape(b,p,n,-1,3)
    #b,p,n,c
    h36m_motion_input=h36m_motion_input-mean
    h36m_motion_input=h36m_motion_input.reshape(b,p,n,-1)#回到b,p,n,c
    
    return h36m_motion_input_,h36m_motion_input,mean



def update_lr_multistep_mine(nb_iter, total_iter, max_lr, min_lr, optimizer) :
    
    # 学习率每 10 个 iter 减为原来的 0.8
    decay_factor = 0.8 ** (nb_iter // 10)  # 每 10 个 iter，乘以 0.8
    current_lr = max_lr * decay_factor  # 从 max_lr 开始递减
    
    # 确保学习率不低于 min_lr
    if current_lr < min_lr:
        current_lr = min_lr
        
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

def gen_velocity(m):
    dm = m[:, :,1:] - m[:, :,:-1]
    return dm

def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m

def seed_set(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def visuaulize_only_person(data,prefix,output_dir,input_len=15,dataset='mocap'):
    for n in range(data.shape[0]):
        #B,P,T,J,K
        data_list=data[n]
        if dataset=='mocap':
            body_edges = np.array(
                [[0, 1], [1, 2], [2, 3], [0, 4],
                [4, 5], [5, 6], [0, 7], [7, 8], [7, 9], [9, 10], [10, 11], [7, 12], [12, 13], [13, 14]]
            )
        else:
            body_edges = np.array(
                [(0, 1), (1, 8), (8, 7), (7, 0),
                (0, 2), (2, 4),
                (1, 3), (3, 5),
                (7, 9), (9, 11),
                (8, 10), (10, 12),
                (6, 7), (6, 8)]
            )

        fig = plt.figure(figsize=(10, 4.5))
        ax = fig.add_subplot(111, projection='3d')

        # 创建保存帧的列表
        frames = []
        frame_names=[]
        length_ = data_list.shape[1]

        for i in range(0, length_):
            ax.cla()  # Clear the previous lines
            ax.grid(False)  # Disable the grid
            ax.set_axis_off()  # Hide the axis
            for j in range(len(data_list)):
                xs = data_list[j, i, :, 0]
                ys = data_list[j, i, :, 1]
                zs = data_list[j, i, :, 2]
                ax.plot(zs, xs, ys, 'y.')
                
                plot_edge = True
                if plot_edge:
                    for edge in body_edges:
                        x = [data_list[j, i, edge[0], 0], data_list[j, i, edge[1], 0]]
                        y = [data_list[j, i, edge[0], 1], data_list[j, i, edge[1], 1]]
                        z = [data_list[j, i, edge[0], 2], data_list[j, i, edge[1], 2]]
                        if i <= input_len:
                            ax.plot(z, x, y, 'green')
                        else:
                            ax.plot(z, x, y, 'blue')
                
                ax.set_xlim3d([-2, 2])
                ax.set_ylim3d([-2, 2])
                ax.set_zlim3d([-0, 2])
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
            
            # 保存当前帧到图像
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            frames.append(imageio.imread(buf))
            buf.close()
            i += 1

        # 保存为GIF
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        imageio.mimsave(f'./{output_dir}/{prefix}_{n}.gif', frames, duration=0.1)


        # 清理临时帧图像
        for frame_filename in frame_names:
            # print(frame_filename)
            os.remove(frame_filename)
        
def visuaulize_only_person_modified(data, prefix, output_dir, input_len=15, dataset='mocap', scale_factors=(1, 1, 1), alpha=1.0):
    for n in range(data.shape[0]):
        # B, P, T, J, K
        data_list = data[n]
        
        if dataset == 'mocap':
            body_edges = np.array(
                [[0, 1], [1, 2], [2, 3], [0, 4],
                 [4, 5], [5, 6], [0, 7], [7, 8], [7, 9], [9, 10], [10, 11], [7, 12], [12, 13], [13, 14]]
            )
        else:
            body_edges = np.array(
                [(0, 1), (1, 8), (8, 7), (7, 0),
                 (0, 2), (2, 4),
                 (1, 3), (3, 5),
                 (7, 9), (9, 11),
                 (8, 10), (10, 12),
                 (6, 7), (6, 8)]
            )

        fig = plt.figure(figsize=(10, 4.5))
        ax = fig.add_subplot(111, projection='3d')

        # 创建保存帧的列表
        frames = []
        frame_names = []
        length_ = data_list.shape[1]

        # 遍历每一帧
        for i in range(0, length_):
            ax.cla()  # 清空之前的绘图
            ax.grid(False)  # 禁用网格
            ax.set_axis_off()  # 隐藏坐标轴

            for j in range(len(data_list)):
                # 根据指定的人物索引，选择缩放因子
                if j == 0:
                    scale = scale_factors[0]
                elif j == 1:
                    scale = scale_factors[1]
                else:
                    scale = scale_factors[2]

                # 对坐标进行缩放
                xs = data_list[j, i, :, 0] * scale
                ys = data_list[j, i, :, 1] * scale
                zs = data_list[j, i, :, 2] * scale

                # 绘制人物
                ax.plot(zs, xs, ys, 'y.', alpha=alpha)  # 使用 alpha 控制透明度

                # 绘制人物骨架
                plot_edge = True
                if plot_edge:
                    for edge in body_edges:
                        x = [data_list[j, i, edge[0], 0] * scale, data_list[j, i, edge[1], 0] * scale]
                        y = [data_list[j, i, edge[0], 1] * scale, data_list[j, i, edge[1], 1] * scale]
                        z = [data_list[j, i, edge[0], 2] * scale, data_list[j, i, edge[1], 2] * scale]
                        if i <= input_len:
                            ax.plot(z, x, y, 'green', alpha=alpha)
                        else:
                            ax.plot(z, x, y, 'blue', alpha=alpha)

                ax.set_xlim3d([-2, 2])
                ax.set_ylim3d([-2, 2])
                ax.set_zlim3d([0, 2])

            # 保存当前帧到图像（PNG格式）
            frame_filename = f"{output_dir}/{prefix}_{n}_frame_{i}.png"
            frame_names.append(frame_filename)
            plt.savefig(frame_filename, format='png', bbox_inches='tight', pad_inches=0)

            # 将当前帧保存为GIF帧
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            frames.append(imageio.imread(buf))
            buf.close()

        # 保存为GIF
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        imageio.mimsave(f'./{output_dir}/{prefix}_{n}.gif', frames, duration=0.1)

        
def visuaulize(data,prefix,output_dir,input_len=15,dataset='mocap'):
    for n in range(data.shape[0]):
        #B,P,T,J,K
        data_list=data[n]
        if dataset=='mocap':
            body_edges = np.array(
                [[0, 1], [1, 2], [2, 3], [0, 4],
                [4, 5], [5, 6], [0, 7], [7, 8], [7, 9], [9, 10], [10, 11], [7, 12], [12, 13], [13, 14]]
            )
        else:
            body_edges = np.array(
                [(0, 1), (1, 8), (8, 7), (7, 0),
                (0, 2), (2, 4),
                (1, 3), (3, 5),
                (7, 9), (9, 11),
                (8, 10), (10, 12),
                (6, 7), (6, 8)]
            )

        fig = plt.figure(figsize=(10, 4.5))
        ax = fig.add_subplot(111, projection='3d')

        # 创建保存帧的列表
        frames = []
        frame_names=[]
        length_ = data_list.shape[1]

        for i in range(0, length_):
            ax.cla()  # Clear the previous lines
            ax.grid(False)  # Disable the grid
            # ax.set_axis_off()  # Hide the axis
            for j in range(len(data_list)):
                xs = data_list[j, i, :, 0]
                ys = data_list[j, i, :, 1]
                zs = data_list[j, i, :, 2]
                ax.plot(zs, xs, ys, 'y.')
                
                plot_edge = True
                if plot_edge:
                    for edge in body_edges:
                        x = [data_list[j, i, edge[0], 0], data_list[j, i, edge[1], 0]]
                        y = [data_list[j, i, edge[0], 1], data_list[j, i, edge[1], 1]]
                        z = [data_list[j, i, edge[0], 2], data_list[j, i, edge[1], 2]]
                        if i <= input_len:
                            ax.plot(z, x, y, 'green')
                        else:
                            ax.plot(z, x, y, 'blue')
                
                ax.set_xlim3d([-2, 2])
                ax.set_ylim3d([-2, 2])
                ax.set_zlim3d([-0, 2])
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
            
            # 保存当前帧到图像
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            frames.append(imageio.imread(buf))
            buf.close()
            i += 1

        # 保存为GIF
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        imageio.mimsave(f'./{output_dir}/{prefix}_{n}.gif', frames, duration=0.1)


        # 清理临时帧图像
        for frame_filename in frame_names:
            # print(frame_filename)
            os.remove(frame_filename)
            
def visuaulize2(data,prefix,output_dir,interval=6,input_len=15,dataset='mocap'):
    data=data[:,:,::interval]
    for n in range(data.shape[0]):
        #B,P,T,J,K
        data_list=data[n]
        if dataset=='mocap':
            body_edges = np.array(
                [[0, 1], [1, 2], [2, 3], [0, 4],
                [4, 5], [5, 6], [0, 7], [7, 8], [7, 9], [9, 10], [10, 11], [7, 12], [12, 13], [13, 14]]
            )
        else:
            body_edges = np.array(
                [(0, 1), (1, 8), (8, 7), (7, 0),
                (0, 2), (2, 4),
                (1, 3), (3, 5),
                (7, 9), (9, 11),
                (8, 10), (10, 12),
                (6, 7), (6, 8)]
            )

        fig = plt.figure(figsize=(10, 4.5))
        ax = fig.add_subplot(111, projection='3d')
        # 创建保存帧的列表
        frames = []
        frame_names=[]
        length_ = data_list.shape[1]#12
        input_len=length_//2#6
        for i in range(0, length_):#0到11
            opaques=np.linspace(0.2, 1.0, length_//2)#6个层次
            opaque=opaques[i%len(opaques)]
            if i%len(opaques)==0:
                ax.cla()  # Clear the previous lines
                ax.set_axis_off() 
            ax.grid(False)  # Disable the grid
            for j in range(len(data_list)):
                xs = data_list[j, i, :, 0]
                ys = data_list[j, i, :, 1]
                zs = data_list[j, i, :, 2]
                ax.plot(zs, xs, ys, 'y.', alpha=opaque)#!修改点的透明度
                
                plot_edge = True
                if plot_edge:
                    for edge in body_edges:
                        x = [data_list[j, i, edge[0], 0], data_list[j, i, edge[1], 0]]
                        y = [data_list[j, i, edge[0], 1], data_list[j, i, edge[1], 1]]
                        z = [data_list[j, i, edge[0], 2], data_list[j, i, edge[1], 2]]
                        if i < input_len:
                            # 设置输入序列边的透明度
                            ax.plot(z, x, y, 'green', alpha=opaque)  # 输入序列使用0.6的透明度
                        else:
                            # 设置预测序列边的透明度
                            ax.plot(z, x, y, 'blue', alpha=opaque)   # 预测序列使用0.8的透明度
                
                ax.set_xlim3d([-2, 2])
                ax.set_ylim3d([-2, 2])
                ax.set_zlim3d([-0, 2])
                # ax.set_xlabel("x")
                # ax.set_ylabel("y")
                # ax.set_zlabel("z")
            
            # 保存当前帧到图像
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            frames.append(imageio.imread(buf))
            buf.close()
            i += 1

        # 保存为GIF
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        imageio.mimsave(f'./{output_dir}/{prefix}_{n}.gif', frames, duration=0.1)


        # 清理临时帧图像
        for frame_filename in frame_names:
            # print(frame_filename)
            os.remove(frame_filename)
            
def visuaulize_bianhao(data,prefix,output_dir,input_len=15,dataset='mocap'):
    for n in range(data.shape[0]):
        #B,P,T,J,K
        data_list=data[n]
        if dataset=='mocap':
            body_edges = np.array(
                [[0, 1], [1, 2], [2, 3], [0, 4],
                [4, 5], [5, 6], [0, 7], [7, 8], [7, 9], [9, 10], [10, 11], [7, 12], [12, 13], [13, 14]]
            )
        else:
            body_edges = np.array(
                [(0, 1), (1, 8), (8, 7), (7, 0),
                (0, 2), (2, 4),
                (1, 3), (3, 5),
                (7, 9), (9, 11),
                (8, 10), (10, 12),
                (6, 7), (6, 8)]
            )

        fig = plt.figure(figsize=(10, 4.5))
        ax = fig.add_subplot(111, projection='3d')

        # 创建保存帧的列表
        frames = []
        frame_names=[]
        length_ = data_list.shape[1]

        for i in range(0, length_):
            ax.cla()  # Clear the previous lines
            ax.grid(False)  # Disable the grid
            for j in range(len(data_list)):
                xs = data_list[j, i, :, 0]
                ys = data_list[j, i, :, 1]
                zs = data_list[j, i, :, 2]
                ax.plot(zs, xs, ys, 'y.')
                # 在每个关节点旁边添加编号
                for k in range(len(xs)):
                    ax.text(zs[k], xs[k], ys[k], f'{k}', color='red', fontsize=8)
                plot_edge = True
                if plot_edge:
                    for edge in body_edges:
                        x = [data_list[j, i, edge[0], 0], data_list[j, i, edge[1], 0]]
                        y = [data_list[j, i, edge[0], 1], data_list[j, i, edge[1], 1]]
                        z = [data_list[j, i, edge[0], 2], data_list[j, i, edge[1], 2]]
                        if i <= input_len:
                            ax.plot(z, x, y, 'green')
                        else:
                            ax.plot(z, x, y, 'blue')
                
                ax.set_xlim3d([-2, 2])
                ax.set_ylim3d([-2, 2])
                ax.set_zlim3d([-0, 2])
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
            
            # 保存当前帧到图像
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            frames.append(imageio.imread(buf))
            buf.close()
            i += 1

        # 保存为GIF
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        imageio.mimsave(f'./{output_dir}/{prefix}_{n}.gif', frames, duration=0.1)


        # 清理临时帧图像
        for frame_filename in frame_names:
            # print(frame_filename)
            os.remove(frame_filename)