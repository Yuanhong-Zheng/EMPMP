from torch.utils.data import Dataset
import numpy as np
import torch
import os
from lib.dataset.ang2joint import *
import networkx as nx

'''
adapted from
https://github.com/wei-mao-2019/HisRepItself/blob/master/utils/amass3d.py
'''

class AMASSDatasets(Dataset):

    def __init__(self, path_to_data, skel_path, input_n=16, output_n=14, skip_rate=2, actions=None, split=0,device='cuda'):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 testing, 2 validation
        :param sample_rate:
        """
        self.path_to_data = os.getcwd()+path_to_data     
        self.split = split
        self.in_n = input_n
        self.out_n = output_n
        # self.sample_rate = opt.sample_rate
        self.p3d = []
        self.keys = []
        self.data_idx = []
        self.joint_used = np.arange(4, 22) # start from 4 for 17 joints, removing the non moving ones
        seq_len = self.in_n + self.out_n

        # # full dataset
        # amass_splits = [
        #     ['MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'EKUT', 'TCD_handMocap', 'ACCAD'],
        #     ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
        #     ['BioMotionLab_NTroje'],
        # ]

        amass_splits = [
            ['CMU'],
            ['HumanEva'],
            ['BioMotionLab_NTroje'],
        ]

        # amass_splits = [
        #     ['HumanEva'],
        #     ['HumanEva'],
        #     ['HumanEva'],
        # ]

        # load mean skeleton
        skel = np.load(skel_path)
        p3d0 = torch.from_numpy(skel['p3d0']).float().to(device)
        parents = skel['parents']
        parent = {}
        for i in range(len(parents)):
            parent[i] = parents[i]
        n = 0

        for ds in amass_splits[split]:
            # print()
            if not os.path.isdir(self.path_to_data + ds):
                print(ds)
                continue
            print('>>> loading {}'.format(ds))
            for sub in os.listdir(self.path_to_data + ds):
                if not os.path.isdir(self.path_to_data + ds + '/' + sub):
                    continue
                for act in os.listdir(self.path_to_data + ds + '/' + sub):
                    if not act.endswith('.npz'):
                        continue
                    # if not ('walk' in act or 'jog' in act or 'run' in act or 'treadmill' in act):
                    #     continue
                    pose_all = np.load(self.path_to_data + ds + '/' + sub + '/' + act)
                    # ['trans', 'gender', 'mocap_framerate', 'betas', 'dmpls', 'poses']
                    # print(pose_all['trans'].shape, pose_all['poses'].shape)
                    # print(pose_all['trans'], pose_all['poses'])
                    # [T, 3], [T, 52*3]
                    # raise ValueError
                    try:
                        trans = pose_all['trans']
                        poses = pose_all['poses']
                    except:
                        print('no poses at {}_{}_{}'.format(ds, sub, act))
                        continue
                    frame_rate = pose_all['mocap_framerate']
                    # gender = pose_all['gender']
                    # dmpls = pose_all['dmpls']
                    # betas = pose_all['betas']
                    # trans = pose_all['trans']
                    fn = poses.shape[0]
                    sample_rate = int(frame_rate // 15)
                    # print(frame_rate, sample_rate)
                    # raise ValueError
                    fidxs = range(0, fn, sample_rate)
                    fn = len(fidxs)
                    poses = poses[fidxs]
                    poses = torch.from_numpy(poses).float().to(device)
                    poses = poses.reshape([fn, -1, 3])
                    # T, 52, 3
                    # remove global rotation
                    # poses[:, 0] = 0
                    # poses += torch.from_numpy(trans[fidxs]).float().to(device)[:, None, :]
                    p3d0_tmp = p3d0.repeat([fn, 1, 1])
                    # p3d = ang2joint(p3d0_tmp, poses, parent)
                    p3d = ang2joint(p3d0_tmp, poses, parent)+torch.from_numpy(trans[fidxs]).float().to(device)[:, None, :]
                    # self.p3d[(ds, sub, act)] = p3d.cpu().data.numpy()
                    self.p3d.append(p3d.cpu().data.numpy())
                    if split  == 2:
                        valid_frames = np.arange(0, fn - seq_len + 1, skip_rate)
                    else:
                        valid_frames = np.arange(0, fn - seq_len + 1, skip_rate)

                    # tmp_data_idx_1 = [(ds, sub, act)] * len(valid_frames)
                    self.keys.append((ds, sub, act))
                    tmp_data_idx_1 = [n] * len(valid_frames)
                    tmp_data_idx_2 = list(valid_frames)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                    n += 1


    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        key, start_frame = self.data_idx[item]
        fs = np.arange(start_frame, start_frame + self.in_n + self.out_n)
        return self.p3d[key][fs]  # , key

    def get_minmax(self):
        joint_to_use = np.array([1, 2, 4, 5, 7, 8, 15, 16, 17, 18, 19, 20, 21])
        min = [1, 1, 1, 1, 1, 1]
        max = [-1, -1, -1, -1, -1, -1]
        for i in range(len(self.p3d)):
            pos = np.array(self.p3d[i])[:, joint_to_use]
            vel = np.zeros_like(pos)
            vel[1:] = pos[1:] - pos[:-1]
            data = np.concatenate((pos, vel), axis=-1)
            min_ = data.min(axis=(0, 1))
            min = np.where(min_ < min, min_, min)
            max_ = data.max(axis=(0, 1))
            max = np.where(max_ > max, max_, max)
        print(min)
        print(max)
        return min, max

def normalize_A(A): # given an adj.matrix, normalize it by multiplying left and right with the degree matrix, in the -1/2 power
        
        A=A+np.eye(A.shape[0])
        
        D=np.sum(A,axis=0)
        
        
        D=np.diag(D.A1)

        
        D_inv = D**-0.5
        D_inv[D_inv==np.infty]=0
        
        return D_inv*A*D_inv

def spatio_temporal_graph(joints_to_consider,temporal_kernel_size,spatial_adjacency_matrix): # given a normalized spatial adj.matrix,creates a spatio-temporal adj.matrix
    number_of_joints=joints_to_consider

    spatio_temporal_adj=np.zeros((temporal_kernel_size,number_of_joints,number_of_joints))
    for t in range(temporal_kernel_size):
        for i in range(number_of_joints):
            spatio_temporal_adj[t,i,i]=1 # create edge between same body joint,for t consecutive frames
            for j in range(number_of_joints):
                if spatial_adjacency_matrix[i,j]!=0: # if the body joints are connected
                    spatio_temporal_adj[t,i,j]=spatial_adjacency_matrix[i,j]
    return spatio_temporal_adj

def get_adj_AMASS(joints_to_consider,temporal_kernel_size): # returns adj.matrix to be fed to the network
    if joints_to_consider==22:
        edgelist = [
                    (0, 1), (0, 2), #(0, 3),
                    (1, 4), (5, 2), #(3, 6),
                    (7, 4), (8, 5), #(6, 9),
                    (7, 10), (8, 11), #(9, 12),
                    #(12, 13), (12, 14),
                    (12, 15),
                    #(13, 16), (12, 16), (14, 17), (12, 17),
                    (12, 16), (12, 17),
                    (16, 18), (19, 17), (20, 18), (21, 19),
                    #(22, 20), #(23, 21),#wrists
                    (1, 16), (2, 17)]

    # create a graph
    G=nx.Graph()
    G.add_edges_from(edgelist)
    # create adjacency matrix
    A = nx.adjacency_matrix(G,nodelist=list(range(0,joints_to_consider))).todense()
    #normalize adjacency matrix
    A=normalize_A(A)
    return torch.Tensor(spatio_temporal_graph(joints_to_consider,temporal_kernel_size,A))