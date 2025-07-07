import torch.utils.data as data
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import os
from io import BytesIO
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from copy import deepcopy as c
import torch_dct as dct
class DATA(Dataset):
    def __init__(self, mode, t_his=15, t_pred=45,use_v=False,n_p=3):
        if mode=="train":
            self.data=np.load('./data/train_3_120_mocap.npy',allow_pickle=True)
        elif mode=="test":
            self.data=np.load('./data/test_3_120_mocap.npy',allow_pickle=True)
        elif mode=="eval_mutpots":
            self.data=np.load('./data/mupots_120_3persons.npy',allow_pickle=True)
        elif mode=="eval_mocap":
            self.data=np.load('./data/test_3_120_mocap.npy',allow_pickle=True)

        # #[n,P,T(120),J,3]
        # self.data=self.data.reshape(self.data.shape[0],self.data.shape[1],self.data.shape[2],-1,3)
        #单人
        if n_p==1:
            self.data=self.data.reshape(self.data.shape[0],self.data.shape[1],self.data.shape[2],-1)[:,:1,:,:]

        self.traj_dim = 15 * 3
        if use_v==True:
            self.traj_dim = 16 * 3
            v = (np.diff(self.data[:,:,:, :1,:], axis=2) * 50).clip(-5.0, 5.0)#计算vel只需要一个关节的信息
            v = np.append(v, v[:,:,-1:], axis=2)
            self.data=np.concatenate((self.data, v), axis=3)
        self.mode = mode
        self.t_his = t_his
        self.t_pred = t_pred
        self.t_total = t_his + t_pred
        self.std, self.mean = None, None
        
        self.normalized = False
        # iterator specific
        self.sample_ind = None
        
    def sample(self):        
        i=np.random.choice(list(range(0,len(self.data))))
        seq=self.data[i]
        traj = seq[:,::2,:]
        if self.t_total!=60:
            start_idx = np.random.choice(range(0, traj.shape[1] - self.t_total))
            traj = traj[:,start_idx:start_idx+self.t_total,:]
        return traj[None, ...]
    


    def sampling_generator(self, num_samples=10000, batch_size=64, aug=True):
        for i in range(num_samples // batch_size):
            sample = []
            for i in range(batch_size):
                sample_i = self.sample()
                sample.append(sample_i)
            sample = np.concatenate(sample, axis=0)
            yield sample[:,:,:self.t_his],sample[:,:,self.t_his:]
            
            
    def iter_generator(self,  batch_size=128):#复现MRT,120帧间隔取60帧
        total_batches = len(self.data) // batch_size
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            data = self.data[start_idx:start_idx + batch_size, :, ::2]
            yield data[:, :, :self.t_his], data[:, :, self.t_his:]
        
        # 处理最后一小部分数据
        if len(self.data) % batch_size != 0:
            start_idx = total_batches * batch_size
            data = self.data[start_idx:, :, ::2]  # 最后的小部分数据
            yield data[:, :, :self.t_his], data[:, :, self.t_his:]
            
    def visuaulize(self,model,prefix,output_dir,cfg):
        with torch.no_grad():
            sample=self.sample()
            # torch.save(sample, 'tensor.pt')
            # sample = torch.load('tensor.pt')
            gt=c(sample)#1,P,T,J,3
            
            input_np=sample[:,:,:cfg.t_his,:,:]
            input=torch.tensor(input_np,dtype=cfg.dtype).to(device=cfg.device)#B,P,t_his,J,3
            results=input[:,:,-1:,:,:].flatten(-2)
            # pad_idx=list(range(cfg.t_his))+[cfg.t_his-1]*cfg.t_pred
            # input_pad=input[:,:,pad_idx,:,:].flatten(-2)#B,P,T,3J
            input_pad=input.flatten(-2)#not really
            input_dct=dct.dct(input_pad)
            valid_lens=None
            output=model(input_dct[:,:,1:15,:]-input_dct[:,:,:14,:],dct.idct(input_dct[:,:,-1:,:]),valid_lens)#B,P,T,3J
            output=dct.idct(output)
            for i in range(1,16):
                results=torch.cat([results,input[:,:,-1:,:,:].flatten(-2)+torch.sum(output[:,:,:i,:],dim=2,keepdim=True)],dim=2)
            results=results[:,:,1:,:].reshape(results.shape[0],results.shape[1],15,cfg.n_joint,-1)
            # output=output.reshape(output.shape[0],output.shape[1],output.shape[2],cfg.n_joint,-1)#B,P,T,J,3
            results=torch.cat((input,results),dim=2)
            results=results.detach().cpu().numpy()

            visuaulize(gt,prefix+"_gt",output_dir)
            visuaulize(results,prefix,output_dir)
def main():
    # 创建训练集实例
    train_dataset = DATA(mode="train", t_his=15, t_pred=45,n_p=3)
    
    print(f"训练集大小: {len(train_dataset)}")
    

    # 使用DataLoader
    generator=train_dataset.iter_generator(batch_size=1)
    for (batch,a) in generator:
        print(f"DataLoader批次形状: {batch.shape}")
        break
    
def visuaulize(data,prefix,output_dir):
    for n in range(data.shape[0]):
        data_list=data[n]
        body_edges = np.array(
            [[0, 1], [1, 2], [2, 3], [0, 4],
            [4, 5], [5, 6], [0, 7], [7, 8], [7, 9], [9, 10], [10, 11], [7, 12], [12, 13], [13, 14]]
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

if __name__ == "__main__":
    main()