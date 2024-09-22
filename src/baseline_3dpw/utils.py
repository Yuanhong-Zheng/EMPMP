import os
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
from io import BytesIO
import random
import torch
from easydict import EasyDict as edict

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
        
def update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer) :
    if nb_iter > 1000000:
        current_lr = 1e-5
    else:
        current_lr = 1e-4

    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

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
    
def visuaulize(data,prefix,output_dir):
    for n in range(data.shape[0]):
        #B,P,T,J,K
        data_list=data[n]
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
                        if i >= 15:
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