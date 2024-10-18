def update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer) :
    if nb_iter > 1000000:
        current_lr = 1e-5
    else:
        current_lr = 3e-4

    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

def update_lr_multistep_mine(nb_iter, total_iter, max_lr, min_lr, optimizer) :
    
    # 学习率每 10 个 iter 减为原来的 0.8
    decay_factor = 0.8 ** (nb_iter // 20)  # 每 10 个 iter，乘以 0.8
    current_lr = max_lr * decay_factor  # 从 max_lr 开始递减
    
    # 确保学习率不低于 min_lr
    if current_lr < min_lr:
        current_lr = min_lr
        
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr