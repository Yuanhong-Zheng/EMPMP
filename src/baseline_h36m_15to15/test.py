import torch
import numpy as np
from src.models_dual_inter_add.utils import predict
def mpjpe_test(config, model, eval_generator,dataset="mocap"):    
    device=config.device
    dct_m=config.dct_m
    idct_m=config.idct_m
    
    model.eval()

    loss_list1=[]
    mpjpe_res=[]
    
    for (h36m_motion_input, h36m_motion_target) in eval_generator:
        h36m_motion_input=torch.tensor(h36m_motion_input,device=device).float()
        h36m_motion_target=torch.tensor(h36m_motion_target,device=device).float()
        if config.deriv_input:
            b,p,n,c = h36m_motion_input.shape
            h36m_motion_input_ = h36m_motion_input.clone()
            #b,p,n,c
            h36m_motion_input_ = torch.matmul(dct_m[:, :, :config.dct_len], h36m_motion_input_.to(device))
        else:
            h36m_motion_input_ = h36m_motion_input.clone()

        motion_pred = model(h36m_motion_input_.to(device))
        motion_pred = torch.matmul(idct_m[:, :config.dct_len, :], motion_pred)#b,p,n,c

        if config.deriv_output:
            offset = h36m_motion_input[:, :,-1:].to(device)#b,p,1,c
            motion_pred = motion_pred[:,:, :config.t_pred] + offset#b,p,n,c
        else:
            motion_pred = motion_pred[:, :config.t_pred]

        b,p,n,c = motion_pred.shape
        motion_pred = motion_pred.reshape(b,p,n,15,3).squeeze(0).cpu().detach()
        h36m_motion_target=h36m_motion_target.reshape(b,p,n,15,3).squeeze(0).cpu().detach()
        if dataset=="mocap":
            loss1=torch.sqrt(((motion_pred/1.8 - h36m_motion_target/1.8) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().tolist()
        else:
            loss1=torch.sqrt(((motion_pred - h36m_motion_target) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().tolist()
        loss_list1.append(np.mean(loss1))#+loss1
        
    mpjpe_res.append(np.mean(loss_list1))
    
    return mpjpe_res

def regress_pred(model,motion_input,config):
    # joint_to_ignore = np.array([16, 20, 23, 24, 28, 31]).astype(np.int64)
    # joint_equal = np.array([13, 19, 22, 13, 27, 30]).astype(np.int64)
        '''
        motion_input:b,p,n,jk
        '''
        outputs = []
        step = config.motion.h36m_target_length_train#10
        
        if step == 45:
            num_step = 1
        else:
            num_step = 45 // step#3
        for idx in range(num_step):
            with torch.no_grad():
                output=predict(model,motion_input,config)
            outputs.append(output)
            motion_input = torch.cat([motion_input[:, :,step:], output], axis=2)
        motion_pred = torch.cat(outputs, axis=2)[:,:,:45]
        
        return motion_pred
    
def mpjpe_test_regress(config, model, eval_generator,dataset="mocap"):    
    device=config.device
    # dct_m=config.dct_m
    # idct_m=config.idct_m
    n_joint=config.n_joint
    
    model.eval()

    loss_list1=[]
    loss_list2=[]
    loss_list3=[]
    mpjpe_res=[]
    
    for (h36m_motion_input, h36m_motion_target) in eval_generator:
        h36m_motion_input=torch.tensor(h36m_motion_input,device=device).float()#b,p,t,jk
        h36m_motion_target=torch.tensor(h36m_motion_target,device=device).float()

        motion_pred = regress_pred(model,h36m_motion_input,config)

        b,p,n,c = motion_pred.shape
        motion_pred = motion_pred.reshape(b,p,n,n_joint,3).cpu().detach()
        h36m_motion_target=h36m_motion_target.reshape(b,p,n,n_joint,3).cpu().detach()#b,p,t,j,3
        
        if dataset=="mocap":
            loss1=torch.sqrt(((motion_pred[:,:,:15]/1.8 - h36m_motion_target[:,:,:15]/1.8) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().astype(np.float64)
            loss2=torch.sqrt(((motion_pred[:,:,:30]/1.8 - h36m_motion_target[:,:,:30]/1.8) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().astype(np.float64)
            loss3=torch.sqrt(((motion_pred[:,:,:45]/1.8 - h36m_motion_target[:,:,:45]/1.8) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().astype(np.float64)
        else:
            loss1=torch.sqrt(((motion_pred[:,:,:15] - h36m_motion_target[:,:,:15]) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().astype(np.float64)
            loss2=torch.sqrt(((motion_pred[:,:,:30] - h36m_motion_target[:,:,:30]) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().astype(np.float64)
            loss3=torch.sqrt(((motion_pred[:,:,:45] - h36m_motion_target[:,:,:45]) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().astype(np.float64)
        loss1=np.mean(loss1,axis=-1).tolist()
        loss2=np.mean(loss2,axis=-1).tolist()
        loss3=np.mean(loss3,axis=-1).tolist()
        
        loss_list1.extend(loss1)
        loss_list2.extend(loss2)
        loss_list3.extend(loss3)
        
    mpjpe_res.append(np.mean(loss_list1))
    mpjpe_res.append(np.mean(loss_list2))
    mpjpe_res.append(np.mean(loss_list3))
    
    return mpjpe_res