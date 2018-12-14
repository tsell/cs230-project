import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
import torch.optim
import numpy as np
from collections import Iterable


dtype = torch.cuda.FloatTensor
#dtype = torch.FloatTensor

def save_np_img(img,filename):
    if(img.shape[0] == 1):
        plt.imshow(np.clip(img[0],0,1))
    else:
        plt.imshow(np.clip(img.transpose(1, 2, 0),0,1))
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def apply_until(net_input,net,n = 100):
    # applies function by funtion of a network
    for i,fun in enumerate(net):
        if i>=n:
            break
        if i==0:
            out = fun(net_input.type(dtype))
        else:
            out = fun(out)
    print(i, "last func. applied:", net[i-1])
    if n == 0:
        return net_input
    else:
        return out


from math import ceil

# given a lists of images as np-arrays, plot them as a row# given 
def plot_image_grid(imgs,nrows=10):
    ncols = ceil( len(imgs)/nrows )
    nrows = min(nrows,len(imgs))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True,figsize=(ncols,nrows),squeeze=False)
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            ax.imshow(imgs[j*nrows+i], cmap='Greys_r', interpolation='none')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    fig.tight_layout(pad=0.1)
    return fig

def save_tensor(out,filename,nrows=8):
    imgs = [img for img in out.data.cpu().numpy()[0]]
    fig = plot_image_grid(imgs,nrows=nrows)
    plt.savefig(filename)
    plt.close()

def show_graph(loss_wrt_noisy, loss_wrt_clean):
    fig = plt.figure(figsize=(7,3), dpi=300)
    
    sp = fig.add_subplot(121)
    sp.plot(loss_wrt_clean)
    sp.set_title('Loss w.r.t clean')
    sp.set_yscale('logit')
    sp.minorticks_off()
    sp.grid(True)
    sp.autoscale_view(tight=True, scaley=True)
    
    sp = fig.add_subplot(122)
    sp.plot(loss_wrt_noisy)
    sp.set_title('Loss w.r.t noisy')
    sp.set_yscale('logit')
    sp.minorticks_off()
    sp.grid(True)
    sp.autoscale_view(tight=True, scaley=True)
    
    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.4,
                wspace=0.35)
    plt.show()