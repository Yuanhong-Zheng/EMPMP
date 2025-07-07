def update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer,model_path=None) :
    if nb_iter > 1000000:
        current_lr = 1e-5
    else:
        current_lr = 3e-4
    if model_path is not None:
        # current_lr = 1e-6
        current_lr = 1e-5
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

def update_lr_multistep_mine(nb_iter, total_iter, max_lr, min_lr, optimizer) :
    
    # Learning rate decreases to 0.8 of original value every 10 iterations
    decay_factor = 0.8 ** (nb_iter // 10)  # Multiply by 0.8 every 10 iterations
    current_lr = max_lr * decay_factor  # Start decreasing from max_lr
    
    # Ensure learning rate is not lower than min_lr
    if current_lr < min_lr:
        current_lr = min_lr
        
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr