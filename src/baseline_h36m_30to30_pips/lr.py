def update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer) :#在这里设置学习率的变化
    if nb_iter > 1000000:
        current_lr = 1e-5
    else:
        current_lr = 3e-4

    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr