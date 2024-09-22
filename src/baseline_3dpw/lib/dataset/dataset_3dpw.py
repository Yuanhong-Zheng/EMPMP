import numpy as np
import os
import torch
from torch.utils.data import DataLoader, ConcatDataset,TensorDataset
from src.baseline_3dpw.lib.dataset.dataset_util import collate_batch, create_dataset,get_datasets_mine
import imageio
import matplotlib.pyplot as plt
from io import BytesIO

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
            
def dataloader_for(dataset, config, **kwargs):
    return DataLoader(dataset,
                      batch_size=config['TRAIN']['batch_size'],
                      num_workers=config['TRAIN']['num_workers'],
                      collate_fn=collate_batch,
                      **kwargs)
    
def dataloader_for_mine(dataset, config, **kwargs):
    return DataLoader(dataset,
                      batch_size=config.batch_size,
                      num_workers=config.num_workers,
                      collate_fn=collate_batch,
                      **kwargs)
def collate_batch_mine(batch):
    """
    将每个batch中的样本合并成一个张量。
    :param batch: list of samples, each sample is a tensor from TensorDataset
    :return: batched tensor
    """
    # 将所有的样本堆叠在一起
    batch=torch.stack([sample[0] for sample in batch])
    padding_mask = torch.ones(batch.size(0), batch.size(1))
    return batch,None, padding_mask
def get_3dpw_dataloader(split,cfg,shuffle,batch_size=None):
    
    if split=="train":
        dataset_train = ConcatDataset(get_datasets_mine(['3dpw'], cfg))
        dataloader_train = dataloader_for_mine(dataset_train, cfg, shuffle=shuffle, pin_memory=True)
        return dataloader_train
    elif split=="test":
        in_F, out_F = cfg.t_his, cfg.t_pred
        dataset_test = create_dataset("3dpw",  split="test", track_size=(in_F+out_F), track_cutoff=in_F, segmented=True)
        dataloader_test = DataLoader(dataset_test, batch_size=cfg.batch_size if batch_size==None else batch_size, num_workers=cfg.num_workers, shuffle=shuffle, collate_fn=collate_batch)
        return dataloader_test
    elif split=="jrt":
        in_F, out_F = cfg.t_his, cfg.t_pred
        
        data=torch.load('data/somof_test.pt',map_location=cfg.device).float()
        data=data.reshape(data.shape[0],data.shape[1],in_F+out_F,-1,3)
        
        dataset_jrt = TensorDataset(data)
        dataloader_jrt = DataLoader(dataset_jrt, batch_size=cfg.batch_size if batch_size is None else batch_size, shuffle=shuffle,collate_fn=collate_batch_mine)
        # print(next(iter(dataloader_jrt)))
        return dataloader_jrt
    
# if __name__=="__main__":    
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--exp_name", type=str, default="", help="Experiment name. Otherwise will use timestamp")
#     parser.add_argument("--cfg", type=str, default="3dpw_test/release.yaml", help="Config name. Otherwise will use default config")
#     parser.add_argument('--dry-run', action='store_true', help="Run just one iteration")
#     args = parser.parse_args()

#     if args.cfg != "":
#         cfg = load_config(os.path.join(os.getcwd(),args.cfg), exp_name=args.exp_name)
#     else:
#         cfg = load_default_config()

#     cfg['dry_run'] = args.dry_run

#     # Set the random seed so operations are deterministic if desired
#     random.seed(cfg['SEED'])
#     torch.manual_seed(cfg['SEED'])
#     np.random.seed(cfg['SEED'])

#     # Compatibility with both gpu and cpu training
#     if torch.cuda.is_available():
#         cfg["DEVICE"] = f"cuda:2"
#     else:
#         cfg["DEVICE"] = "cpu"

#     logger = create_logger(cfg["OUTPUT"]["log_dir"])

#     logger.info("Hello!")
#     logger.info("Initializing with config:")
#     logger.info(cfg)

#     dataset_train = ConcatDataset(get_datasets(['3dpw'], cfg))
#     dataloader_train = dataloader_for(dataset_train, cfg, shuffle=True, pin_memory=True)

#     # #可视化训练集
#     # dataiter = iter(dataloader_train)
#     # joints, masks, padding_mask = next(dataiter)
#     # joints=joints[:5].cpu().detach().numpy()
#     # for i in range(joints.shape[0]):
#     #     visuaulize(joints,'3dpw','3dpw_test/3dpw_vis')

#     in_F, out_F = cfg['TRAIN']['input_track_size'], cfg['TRAIN']['output_track_size']
#     dataset_test = create_dataset("3dpw",  split="test", track_size=(in_F+out_F), track_cutoff=in_F, segmented=True)
#     dataloader_test = DataLoader(dataset_test, batch_size=cfg['TRAIN']['batch_size'], num_workers=cfg['TRAIN']['num_workers'], shuffle=True, collate_fn=collate_batch)

#     # #可视化测试集
#     # dataiter = iter(dataloader_test)
#     # joints, masks, padding_mask = next(dataiter)
#     # joints=joints[:5].cpu().detach().numpy()
#     # for i in range(joints.shape[0]):
#     #     visuaulize(joints,'3dpw_test','3dpw_test/3dpw_vis')

