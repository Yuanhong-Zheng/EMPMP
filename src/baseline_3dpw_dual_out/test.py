import torch
import numpy as np
from src.models_inter_dual_out.utils import AverageMeter,predict
from metrics import VIM, VAM

def random_pred(config, model,iter):
    device=config.device
    dct_m=config.dct_m
    idct_m=config.idct_m

    model.eval()
    
    has_data = any(True for _ in iter)
    if not has_data:
        return None,None
    
    joints, masks, padding_mask=next(iter)
    
    h36m_motion_input=joints[:,:,:config.t_his].flatten(-2)#16
    h36m_motion_target=joints[:,:,config.t_his:].flatten(-2)#14
    
    h36m_motion_input=torch.tensor(h36m_motion_input,device=device).float()
    h36m_motion_target=torch.tensor(h36m_motion_target,device=device).float()
    
    motion_pred,_=predict(model,h36m_motion_input,config)
    
    return h36m_motion_input[:1],motion_pred[:1]

def random_pred_pretrain(config, model,iter,joint_to_use):
    device=config.device
    dct_m=config.dct_m
    idct_m=config.idct_m

    model.eval()
    
    has_data = any(True for _ in iter)
    if not has_data:
        return None,None
    
    data=next(iter)
    data = data.float().to(device)
    batch_size = data.shape[0]
    
    data_to_use = data[:, :, joint_to_use].contiguous().view(batch_size//2, 2, 30, 13, 3)
    batch_size = data_to_use.shape[0]
    
    input_total = data_to_use.permute(0, 1, 3, 2, 4).contiguous()#B,P,J,T,K
    padding_mask=torch.ones(batch_size,2).float().to(device).bool()
    
    input_total[:,:,:,:,[1,2]]=input_total[:,:,:,:,[2,1]]#B,P,J,T,K
    input_total=input_total.transpose(2,3)#B,P,T,J,K
    
    h36m_motion_input=input_total[:,:,:config.t_his].flatten(-2)#16
    h36m_motion_target=input_total[:,:,config.t_his:].flatten(-2)#14
    
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

    return h36m_motion_input[:1],motion_pred[:1]

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
        device=config.device
        dct_m=config.dct_m
        idct_m=config.idct_m
        dct_len=config.dct_len
        
        b,p,n,c = motion_input.shape
        # num_samples += b

        outputs = []
        step = config.motion.h36m_target_length_train#10
        
        if step == 45:
            num_step = 1
        else:
            num_step = 45 // step#3
        for idx in range(num_step):
            with torch.no_grad():
                if config.deriv_input:
                    motion_input_ = motion_input.clone()
                    motion_input_ = torch.matmul(dct_m[:, :, :dct_len], motion_input_.to(device))
                else:
                    motion_input_ = motion_input.clone()
                    
                output = model(motion_input_)
                output = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], output)[:, :,:step, :]
                
                if config.deriv_output:
                    output = output + motion_input[:,:, -1:].repeat(1,1,step,1)#b,p,n,c

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
        h36m_motion_input=torch.tensor(h36m_motion_input,device=device).float()
        h36m_motion_target=torch.tensor(h36m_motion_target,device=device).float()

        motion_pred = regress_pred(model,h36m_motion_input,config)

        b,p,n,c = motion_pred.shape
        motion_pred = motion_pred.reshape(b,p,n,n_joint,3).squeeze(0).cpu().detach()
        h36m_motion_target=h36m_motion_target.reshape(b,p,n,n_joint,3).squeeze(0).cpu().detach()
        
        if dataset=="mocap":
            loss1=torch.sqrt(((motion_pred[:,:15]/1.8 - h36m_motion_target[:,:15]/1.8) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().tolist()
            loss2=torch.sqrt(((motion_pred[:,:30]/1.8 - h36m_motion_target[:,:30]/1.8) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().tolist()
            loss3=torch.sqrt(((motion_pred[:,:45]/1.8 - h36m_motion_target[:,:45]/1.8) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().tolist()
        else:
            loss1=torch.sqrt(((motion_pred[:,:15] - h36m_motion_target[:,:15]) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().tolist()
            loss2=torch.sqrt(((motion_pred[:,:30] - h36m_motion_target[:,:30]) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().tolist()
            loss3=torch.sqrt(((motion_pred[:,:45] - h36m_motion_target[:,:45]) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().tolist()
        
        loss_list1.append(np.mean(loss1))#+loss1
        loss_list2.append(np.mean(loss2))#+loss2
        loss_list3.append(np.mean(loss3))#+loss3
        
    mpjpe_res.append(np.mean(loss_list1))
    mpjpe_res.append(np.mean(loss_list2))
    mpjpe_res.append(np.mean(loss_list3))
    
    return mpjpe_res

def vim_test(config, model, eval_generator,dataset="3dpw",return_all=True,select_frames=[1, 3, 7, 9, 13]):    
    device=config.device
    dct_m=config.dct_m
    idct_m=config.idct_m
    
    
    vim_avg = AverageMeter()
    
    model.eval()

    
    for (joints, masks, padding_mask) in eval_generator:
        h36m_motion_input=joints[:,:,:config.t_his].flatten(-2)#16
        h36m_motion_target=joints[:,:,config.t_his:].flatten(-2)#14
        
        h36m_motion_input=torch.tensor(h36m_motion_input,device=device).float()
        h36m_motion_target=torch.tensor(h36m_motion_target,device=device).float()
        
        motion_pred,_=predict(model,h36m_motion_input,config)
        #目标：b,n,p*c
        b,p,n,c = motion_pred.shape
        motion_pred = motion_pred.transpose(1,2).flatten(-2).squeeze(0).cpu().detach().numpy()
        h36m_motion_target=h36m_motion_target.transpose(1,2).flatten(-2).squeeze(0).cpu().detach().numpy()
        
        for person in range(p):
            # if person==1:
            #     print("跳过第二个人")
            #     continue
            JK=c
            J=config.n_joint
            K=JK//J
            
            for k in range(len(h36m_motion_target)):#k是样本索引
                if padding_mask[k,person]==0:
                    continue
                
                person_out_joints = h36m_motion_target[k,:,JK*person:JK*(person+1)]
                assert person_out_joints.shape == (n, J*K)
                person_pred_joints = motion_pred[k,:,JK*person:JK*(person+1)]
                person_masks = np.ones((n, J))
                
                vim_score = VIM(person_out_joints, person_pred_joints, dataset,  person_masks) * 100 # *100 for 3dpw

            
                if return_all:
                    vim_avg.update(vim_score, 1)
                else:
                    vim_100 = vim_score[2]
                    vim_avg.update(vim_100, 1)
    
    return vim_avg.avg[select_frames]

def vim_test_pretrain(config, model, test_loader,joint_to_use,dataset="3dpw",return_all=True,select_frames=[1, 3, 7, 9, 13]):    
    device=config.device
    dct_m=config.dct_m
    idct_m=config.idct_m
    
    vim_avg = AverageMeter()
    
    model.eval()

    for i, data in enumerate(test_loader):
        data = data.float().to(device)
        batch_size = data.shape[0]
        if batch_size % 2 != 0:
            continue
        data_to_use = data[:, :, joint_to_use].contiguous().view(batch_size//2, 2, 30, 13, 3)
        batch_size = data_to_use.shape[0]
        
        input_total = data_to_use.permute(0, 1, 3, 2, 4).contiguous()#B,P,J,T,K
        padding_mask=torch.ones(batch_size,2).float().to(device).bool()
        
        input_total[:,:,:,:,[1,2]]=input_total[:,:,:,:,[2,1]]#B,P,J,T,K
        input_total=input_total.transpose(2,3)#B,P,T,J,K
        
        h36m_motion_input=input_total[:,:,:config.t_his].flatten(-2)#16
        h36m_motion_target=input_total[:,:,config.t_his:].flatten(-2)#14

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
        #目标：b,n,p*c
        b,p,n,c = motion_pred.shape
        motion_pred = motion_pred.transpose(1,2).flatten(-2).squeeze(0).cpu().detach().numpy()
        h36m_motion_target=h36m_motion_target.transpose(1,2).flatten(-2).squeeze(0).cpu().detach().numpy()
        
        for person in range(p):
            if person==1:
                # print("跳过第二个人")
                continue
            JK=c
            J=config.n_joint
            K=JK//J
            
            for k in range(len(h36m_motion_target)):#k是样本索引
                if padding_mask[k,person]==0:
                    continue
                
                person_out_joints = h36m_motion_target[k,:,JK*person:JK*(person+1)]
                assert person_out_joints.shape == (n, J*K)
                person_pred_joints = motion_pred[k,:,JK*person:JK*(person+1)]
                person_masks = np.ones((n, J))
                
                vim_score = VIM(person_out_joints, person_pred_joints, dataset,  person_masks) * 100 # *100 for 3dpw

            
                if return_all:
                    vim_avg.update(vim_score, 1)
                else:
                    vim_100 = vim_score[2]
                    vim_avg.update(vim_100, 1)
    
    return vim_avg.avg[select_frames]