from torch.autograd import Variable
import torch
import torch.optim
import copy
import numpy as np
from scipy.linalg import hadamard

from .helpers import *

dtype = torch.cuda.FloatTensor
#dtype = torch.FloatTensor
           

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.5**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer



def fit(net,
        img_noisy_var,
        num_channels,
        img_clean_var,
        num_iter = 5000,
        LR = 0.01,
        OPTIMIZER='adam',
        opt_input = False,
        reg_noise_std = 0,
        reg_noise_decayevery = 100000,
        mask_var = None,
        apply_f = None,
        loss_func = torch.nn.MSELoss(),
        mse_loss_weight = 0.0,
        decaylr = False,
        net_input = None,
        net_input_gen = "random",
        find_best=False,
       ):

    if net_input is not None:
        print("input provided")
    else:
        # Calculate size of network input
        totalupsample = 2**len(num_channels)
        width = int(img_clean_var.data.shape[2]/totalupsample)
        height = int(img_clean_var.data.shape[3]/totalupsample)
        shape = [1, num_channels[0], width, height]
        print("network input shape: ", shape)

        # Feed uniform noise into the network 
        net_input = Variable(torch.zeros(shape))
        net_input.data.uniform_()
        net_input.data *= 1./10
        
    net_input_saved = net_input.data.clone()
    noise = net_input.data.clone()
        
    if net_input_gen == "downsample":
        # Add the downsampled image to the noise
        down = np_to_pil(var_to_np(img_noisy_var))
        down = down.resize((width, height), Image.ANTIALIAS)
        down = pil_to_np(down)
        repetitions = shape[1] / down.shape[0]
        remainder = shape[1] % down.shape[0]
        down = np.repeat(down, repetitions, axis=0)
        down = np.append(down, down[:remainder,:,:], axis=0)
        np.random.shuffle(down)
        down = np_to_var(down)
        print("downsampled image shape: ", down.shape)
        net_input_saved = net_input.data.clone()
        net_input.data += down + -0.05

    p = [x for x in net.parameters()]

    if(opt_input == True):
        net_input.requires_grad = True
        p += [net_input]

    loss_wrt_noisy = np.zeros(num_iter)
    loss_wrt_truth = np.zeros(num_iter)

    if OPTIMIZER == 'SGD':
        print("optimize with SGD", LR)
        optimizer = torch.optim.SGD(p, lr=LR,momentum=0.9)
    elif OPTIMIZER == 'adam':
        print("optimize with adam", LR)
        optimizer = torch.optim.Adam(p, lr=LR)

    noise_energy = torch.nn.MSELoss()(img_noisy_var, img_clean_var)

    if find_best:
        best_net = copy.deepcopy(net)
        best_loss = 1000000.0
        
    loss_goal = img_noisy_var
    if mask_var is not None:
        loss_goal = img_noisy_var * mask_var

    for i in range(num_iter):
        if decaylr is True:
            optimizer = exp_lr_scheduler(optimizer, i, init_lr=LR, lr_decay_epoch=100)
        if reg_noise_std > 0:
            if i % reg_noise_decayevery == 0:
                reg_noise_std *= 0.7
            net_input = Variable(net_input_saved + (noise.normal_() * reg_noise_std))
        optimizer.zero_grad()
        out = net(net_input.type(dtype))

        # training loss 
        loss_actual = out
        if mask_var is not None:
                loss_actual = out * mask_var
        elif apply_f:
                loss_actual = apply_f(out)
                
        if loss_func:
            # The loss we optimize is formed by comparing the current output to the noisy image.
            loss = loss_func(loss_actual, loss_goal) + mse_loss_weight * torch.nn.MSELoss()(loss_actual, loss_goal)
            # true_loss is formed by comparing the current output to the "clean", un-noisy image.
            true_loss = loss_func(Variable(out.data, requires_grad=False), img_clean_var)
            true_loss = true_loss + mse_loss_weight * torch.nn.MSELoss()(Variable(out.data, requires_grad=False), img_clean_var)
        else:
            loss = torch.nn.MSELoss()(loss_actual, loss_goal)
            true_loss = torch.nn.MSELoss()(Variable(out.data, requires_grad=False), img_clean_var)
        
        loss.backward()
        loss_wrt_noisy[i] = loss.data.cpu().numpy()
        loss_wrt_truth[i] = true_loss.data.cpu().numpy()
        
        # Every ten iterations, output the statistics.
        if i % 10 == 0:
            # loss2 uses the original network input, without the reg_noise added.
            out2 = net(Variable(net_input_saved).type(dtype))
            loss2 = loss_func(out2, img_clean_var) + mse_loss_weight * torch.nn.MSELoss()(out2, img_clean_var)
            print ('Iteration %05d   Train loss %f  Actual loss %f Actual loss orig %f  Noise Energy %f' %
                   (i, loss.data[0],true_loss.data[0],loss2.data[0],noise_energy.data[0]), '\r', end='')
        
        if find_best:
            # if training loss improves by at least one percent, we found a new best net
            if best_loss > 1.005*loss.data[0]:
                best_loss = loss.data[0]
                best_net = copy.deepcopy(net)

        optimizer.step()
    if find_best:
        net = best_net
    return loss_wrt_noisy, loss_wrt_truth, net_input_saved, net

