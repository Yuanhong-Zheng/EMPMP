import torch
import numpy as np
from src.models_dual_inter_traj_3dpw.utils import AverageMeter,predict,Get_RC_Data
from metrics import VIM, VAM
from src.models_dual_inter_traj_3dpw.utils import Get_RC_Data,visuaulize,paixu_person
from src.models_dual_inter_traj_3dpw.model import inverse_sort_tensor
import time

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
    if config.rc:
        h36m_motion_input,h36m_motion_target=Get_RC_Data(h36m_motion_input,h36m_motion_target)
    motion_pred=predict(model,h36m_motion_input,config,h36m_motion_target=h36m_motion_target)
    
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
    #!摆正数据
    # input_total[:,:,:,:,[1,2]]=input_total[:,:,:,:,[2,1]]#B,P,J,T,K
    
    input_total=input_total.transpose(2,3)#B,P,T,J,K
    
    h36m_motion_input=input_total[:,:,:config.t_his].flatten(-2)#16
    h36m_motion_target=input_total[:,:,config.t_his:].flatten(-2)#14
    
    h36m_motion_input=torch.tensor(h36m_motion_input,device=device).float()
    h36m_motion_target=torch.tensor(h36m_motion_target,device=device).float()
    
    motion_pred=predict(model,h36m_motion_input,config)

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




    # return vim_score

# def cal_mpjpe(config, model):    
#     device=config.device
#     # dct_m=config.dct_m
#     # idct_m=config.idct_m
#     n_joint=config.n_joint
    
#     model.eval()

#     loss_list1=[]
#     loss_list2=[]
#     loss_list3=[]
#     mpjpe_res=[]
    
#     for (h36m_motion_input, h36m_motion_target) in eval_generator:
#         h36m_motion_input=torch.tensor(h36m_motion_input,device=device).float()#b,p,t,jk
#         h36m_motion_target=torch.tensor(h36m_motion_target,device=device).float()

#         motion_pred = regress_pred(model,h36m_motion_input,config)#预测结果

#         b,p,n,c = motion_pred.shape
#         motion_pred = motion_pred.reshape(b,p,n,n_joint,3).cpu().detach()
#         h36m_motion_target=h36m_motion_target.reshape(b,p,n,n_joint,3).cpu().detach()#b,p,t,j,3
        
#         if dataset=="mocap":
#             loss1=torch.sqrt(((motion_pred[:,:,:10]/1.8 - h36m_motion_target[:,:,:10]/1.8) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().astype(np.float64)
#             loss2=torch.sqrt(((motion_pred[:,:,:20]/1.8 - h36m_motion_target[:,:,:20]/1.8) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().astype(np.float64)
#             loss3=torch.sqrt(((motion_pred[:,:,:30]/1.8 - h36m_motion_target[:,:,:30]/1.8) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().astype(np.float64)
#         else:#mupots数据集
#             loss1=torch.sqrt(((motion_pred[:,:,:10] - h36m_motion_target[:,:,:10]) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().astype(np.float64)
#             loss2=torch.sqrt(((motion_pred[:,:,:20] - h36m_motion_target[:,:,:20]) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().astype(np.float64)
#             loss3=torch.sqrt(((motion_pred[:,:,:30] - h36m_motion_target[:,:,:30]) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().astype(np.float64)
#         loss1=np.mean(loss1,axis=-1).tolist()
#         loss2=np.mean(loss2,axis=-1).tolist()
#         loss3=np.mean(loss3,axis=-1).tolist()
        
#         loss_list1.extend(loss1)
#         loss_list2.extend(loss2)
#         loss_list3.extend(loss3)
        
#     mpjpe_res.append(np.mean(loss_list1))
#     mpjpe_res.append(np.mean(loss_list2))
#     mpjpe_res.append(np.mean(loss_list3))
    
#     return mpjpe_res

def mpjpe_vim_test(config, model, eval_generator,is_mocap,select_vim_frames=[1, 3, 7, 9, 13],select_mpjpe_frames=[10,20,30]):    
    def cal_vim(motion_pred,h36m_motion_target,vim_avg):    
        #目标：b,n,p*c
        b,p,n,c = motion_pred.shape
        motion_pred = motion_pred.transpose(1,2).flatten(-2).squeeze(0).cpu().detach().numpy()
        h36m_motion_target=h36m_motion_target.transpose(1,2).flatten(-2).squeeze(0).cpu().detach().numpy()
        
        for person in range(p):
            # if person==1:
            #     print("跳过第二个人")
            #     continue
            JK=c
            K=3
            J=c//K
            i=0
            for k in range(len(h36m_motion_target)):#k是样本索引
                person_out_joints = h36m_motion_target[k,:,JK*person:JK*(person+1)]
                assert person_out_joints.shape == (n, J*K)
                person_pred_joints = motion_pred[k,:,JK*person:JK*(person+1)]
                person_masks = np.ones((n, J))
                
                vim_score = VIM(person_out_joints, person_pred_joints) * 100 # *100 for 3dpw
                vim_avg.update(vim_score, 1)
    def cal_mpjpe(motion_pred,h36m_motion_target,is_mocap,select_frames=[10,20,30]):    
        b,p,n,c = motion_pred.shape
        n_joint=c//3
        motion_pred = motion_pred.reshape(b,p,n,n_joint,3).cpu().detach()
        h36m_motion_target=h36m_motion_target.reshape(b,p,n,n_joint,3).cpu().detach()#b,p,t,j,3
        
        frame1=select_frames[0]
        frame2=select_frames[1]
        frame3=select_frames[2]
        if is_mocap:
            loss1=torch.sqrt(((motion_pred[:,:,:frame1]/1.8 - h36m_motion_target[:,:,:frame1]/1.8) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().astype(np.float64)
            loss2=torch.sqrt(((motion_pred[:,:,:frame2]/1.8 - h36m_motion_target[:,:,:frame2]/1.8) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().astype(np.float64)
            loss3=torch.sqrt(((motion_pred[:,:,:frame3]/1.8 - h36m_motion_target[:,:,:frame3]/1.8) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().astype(np.float64)
        else: # mupots数据集or 3dpw数据集
            loss1=torch.sqrt(((motion_pred[:,:,:frame1] - h36m_motion_target[:,:,:frame1]) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().astype(np.float64)
            loss2=torch.sqrt(((motion_pred[:,:,:frame2] - h36m_motion_target[:,:,:frame2]) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().astype(np.float64)
            loss3=torch.sqrt(((motion_pred[:,:,:frame3] - h36m_motion_target[:,:,:frame3]) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).numpy().astype(np.float64)
        loss1=np.mean(loss1,axis=-1).tolist()
        loss2=np.mean(loss2,axis=-1).tolist()
        loss3=np.mean(loss3,axis=-1).tolist()
        
        return loss1,loss2,loss3
    def APE(V_pred, V_trgt, frame_idx):
        V_pred = V_pred - V_pred[:, :, :, 0:1, :]
        V_trgt = V_trgt - V_trgt[:, :, :, 0:1, :]
        scale = 1000
        err = np.arange(len(frame_idx), dtype=np.float_)
        for idx in range(len(frame_idx)):
            err[idx] = torch.mean(torch.mean(torch.norm(V_trgt[:, :, frame_idx[idx]-1, :, :] - V_pred[:, :, frame_idx[idx]-1, :, :], dim=3), dim=2),dim=1).cpu().data.numpy().mean()
        return err * scale

    def JPE(V_pred, V_trgt, frame_idx):
        scale = 1000
        err = np.arange(len(frame_idx), dtype=np.float_)
        for idx in range(len(frame_idx)):
            err[idx] = torch.mean(torch.mean(torch.norm(V_trgt[:, :, frame_idx[idx]-1, :, :] - V_pred[:, :, frame_idx[idx]-1, :, :], dim=3), dim=2), dim=1).cpu().data.numpy().mean()
        return err * scale

    def FDE(V_pred,V_trgt, frame_idx):
        scale = 1000
        err = np.arange(len(frame_idx), dtype=np.float_)
        for idx in range(len(frame_idx)):
            err[idx] = torch.linalg.norm(V_trgt[:, :, frame_idx[idx]-1:frame_idx[idx], : 1, :] - V_pred[:, :, frame_idx[idx]-1:frame_idx[idx], : 1, :], dim=-1).mean(1).mean()
        return err * scale
    
    def cal_jpe(motion_pred,h36m_motion_target,is_mocap,select_frames=[10,20,30]):    
        b,p,n,c = motion_pred.shape
        n_joint=c//3
        motion_pred = motion_pred.reshape(b,p,n,n_joint,3).cpu().detach()
        h36m_motion_target=h36m_motion_target.reshape(b,p,n,n_joint,3).cpu().detach()#b,p,t,j,3
        
        jpe=JPE(motion_pred,h36m_motion_target,select_frames)
        if is_mocap:
            jpe=jpe/1.8

        return jpe
    def cal_ape(motion_pred,h36m_motion_target,is_mocap,select_frames=[10,20,30]):    
        b,p,n,c = motion_pred.shape
        n_joint=c//3
        motion_pred = motion_pred.reshape(b,p,n,n_joint,3).cpu().detach()
        h36m_motion_target=h36m_motion_target.reshape(b,p,n,n_joint,3).cpu().detach()#b,p,t,j,3
        
        ape=APE(motion_pred,h36m_motion_target,select_frames)
        if is_mocap:
            ape=ape/1.8

        return ape
    def cal_fde(motion_pred,h36m_motion_target,is_mocap,select_frames=[10,20,30]):    
        b,p,n,c = motion_pred.shape
        n_joint=c//3
        motion_pred = motion_pred.reshape(b,p,n,n_joint,3).cpu().detach()
        h36m_motion_target=h36m_motion_target.reshape(b,p,n,n_joint,3).cpu().detach()#b,p,t,j,3
        
        fde=FDE(motion_pred,h36m_motion_target,select_frames)
        if is_mocap:
            fde=fde/1.8

        return fde
    
    device=config.device
    
    vim_avg = AverageMeter()
    
    loss_list1=[]
    loss_list2=[]
    loss_list3=[]
    mpjpe_res=[]
    jpe_res=[]
    ape_res=[]
    fde_res=[]
    
    model.eval()
    
    # joint_list=[]
    # for (joints, masks, padding_mask) in eval_generator:
    #     h36m_motion_input=joints[:,:,:config.t_his].flatten(-2)#16
    #     h36m_motion_target=joints[:,:,config.t_his:].flatten(-2)#14
            
    #     h36m_motion_input=torch.tensor(h36m_motion_input,device=device).float()
    #     h36m_motion_target=torch.tensor(h36m_motion_target,device=device).float()
        
    #     h36m_motion_input,h36m_motion_target=Get_RC_Data(h36m_motion_input,h36m_motion_target)
    #     joints=torch.cat([h36m_motion_input,h36m_motion_target],dim=2).reshape(-1,2,30,13,3)
    #     joint_list.append(joints)
    # import pdb;pdb.set_trace()
    # joint_list=torch.cat(joint_list,dim=0)
    # torch.save(joint_list,'somof_test_rc.pt',_use_new_zipfile_serialization=False)
    # start_time=time.time()
    # batch_num=0
    for (joints, masks, padding_mask) in eval_generator:
        # batch_num+=1
        h36m_motion_input=joints[:,:,:config.t_his].flatten(-2)#16
        h36m_motion_target=joints[:,:,config.t_his:].flatten(-2)#14
            
        h36m_motion_input=torch.tensor(h36m_motion_input,device=device).float()
        h36m_motion_target=torch.tensor(h36m_motion_target,device=device).float()
        if config.paixu:#DCT之前
            h36m_motion_input,h36m_motion_target,sorted_indices=paixu_person(h36m_motion_input,h36m_motion_target)
        if config.rc:
            h36m_motion_input,h36m_motion_target=Get_RC_Data(h36m_motion_input,h36m_motion_target)
        motion_pred=predict(model,h36m_motion_input,config,h36m_motion_target=h36m_motion_target)
        if config.paixu:
            motion_pred=inverse_sort_tensor(motion_pred,sorted_indices)
            h36m_motion_target=inverse_sort_tensor(h36m_motion_target,sorted_indices)
        cal_vim(motion_pred,h36m_motion_target,vim_avg)
        loss1,loss2,loss3=cal_mpjpe(motion_pred,h36m_motion_target,is_mocap=is_mocap,select_frames=select_mpjpe_frames)
        jpe=cal_jpe(motion_pred,h36m_motion_target,is_mocap=is_mocap,select_frames=select_mpjpe_frames)
        ape=cal_ape(motion_pred,h36m_motion_target,is_mocap=is_mocap,select_frames=select_mpjpe_frames)
        fde=cal_fde(motion_pred,h36m_motion_target,is_mocap=is_mocap,select_frames=select_mpjpe_frames)
        
        
        loss_list1.extend(loss1)
        loss_list2.extend(loss2)
        loss_list3.extend(loss3)
        
        jpe_res.append(jpe)
        ape_res.append(ape)
        fde_res.append(fde)
    # end_time=time.time()
    # print('time_per_batch:********************',(end_time-start_time)/batch_num)
    mpjpe_res.append(np.mean(loss_list1))
    mpjpe_res.append(np.mean(loss_list2))
    mpjpe_res.append(np.mean(loss_list3))
    mpjpe_res=np.array(mpjpe_res)
    
    jpe_res=np.array(jpe_res)
    jpe_res=np.mean(jpe_res,axis=0)
    
    ape_res=np.array(ape_res)
    ape_res=np.mean(ape_res,axis=0)
    
    fde_res=np.array(fde_res)
    fde_res=np.mean(fde_res,axis=0)
    return mpjpe_res,vim_avg.avg[select_vim_frames]/1.8 if is_mocap else vim_avg.avg[select_vim_frames],jpe_res,ape_res,fde_res

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
        if config.rc:
            h36m_motion_input,h36m_motion_target=Get_RC_Data(h36m_motion_input,h36m_motion_target)
        motion_pred=predict(model,h36m_motion_input,config,h36m_motion_target=h36m_motion_target)
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
                    print("*******************************************")
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

def all_vis(config, model, eval_generator,dataset="3dpw",return_all=True,select_frames=[1, 3, 7, 9, 13]):    
    device=config.device
    dct_m=config.dct_m
    idct_m=config.idct_m

    model.eval()

    sample=0
    for (joints, masks, padding_mask) in eval_generator:
        sample+=1
        if sample!=81:
            continue
        print(f"sample:{sample}")
        
        h36m_motion_input=joints[:,:,:config.t_his].flatten(-2)#16
        h36m_motion_target=joints[:,:,config.t_his:].flatten(-2)#14
        
        h36m_motion_input=torch.tensor(h36m_motion_input,device=device).float()
        h36m_motion_target=torch.tensor(h36m_motion_target,device=device).float()
        if config.rc:
            h36m_motion_input,h36m_motion_target=Get_RC_Data(h36m_motion_input,h36m_motion_target)
        motion_pred=predict(model,h36m_motion_input,config,h36m_motion_target=h36m_motion_target)
        #目标：b,n,p*c
        b,p,n,c = motion_pred.shape
        motion_pred = motion_pred.reshape(b,p,n,config.n_joint,3)
        h36m_motion_target=h36m_motion_target.reshape(b,p,n,config.n_joint,3)
        h36m_motion_input=h36m_motion_input.reshape(b,p,config.t_his,config.n_joint,3)
        
        pred=torch.cat([h36m_motion_input,motion_pred],dim=2).cpu().detach().numpy()
        gt=torch.cat([h36m_motion_input,h36m_motion_target],dim=2).cpu().detach().numpy()
        
        np.save(f"{config.vis_dir}/sample{sample}_pred.npy",pred)
        # np.save(f"{config.vis_dir}/sample{sample}_gt.npy",gt)
        # break
        visuaulize(pred,f"sample{sample}",config.vis_dir,input_len=15,dataset='mupots')
        # visuaulize(gt,f"gt_sample{sample}",config.vis_dir,input_len=15,dataset='mupots')
        


def vim_test_pretrain(config, model, test_loader,joint_to_use,dataset="3dpw",return_all=True,select_frames=[1, 3, 7, 9, 13]):  
    # print('begin vim test')  
    
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
        #!摆正数据
        # input_total[:,:,:,:,[1,2]]=input_total[:,:,:,:,[2,1]]#B,P,J,T,K
        
        input_total=input_total.transpose(2,3)#B,P,T,J,K
        
        h36m_motion_input=input_total[:,:,:config.t_his].flatten(-2)#16
        h36m_motion_target=input_total[:,:,config.t_his:].flatten(-2)#14

        h36m_motion_input=torch.tensor(h36m_motion_input,device=device).float()
        h36m_motion_target=torch.tensor(h36m_motion_target,device=device).float()
        
        motion_pred=predict(model,h36m_motion_input,config)

        #目标：b,n,p*c
        b,p,n,c = motion_pred.shape
        motion_pred = motion_pred.transpose(1,2).flatten(-2).squeeze(0).cpu().detach().numpy()
        h36m_motion_target=h36m_motion_target.transpose(1,2).flatten(-2).squeeze(0).cpu().detach().numpy()
        
        for person in range(p):
            # if person==1:
            #     # print("跳过第二个人")
            #     continue
            JK=c
            J=config.n_joint
            K=JK//J
            
            for k in range(len(h36m_motion_target)):#k是样本索引
                if padding_mask[k,person]==0:
                    print("*********************")
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
    # print('end vim test')
    return vim_avg.avg[select_frames]