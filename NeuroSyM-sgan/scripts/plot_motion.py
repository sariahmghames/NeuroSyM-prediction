import os
import argparse
import time
import torch
import numpy as np
import inspect
from contextlib import contextmanager
import subprocess
import matplotlib.cm
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation
import imageio.v2 as imageio


parser = argparse.ArgumentParser()
parser.add_argument('--traj_path', type=str)




def plot_motion(trajectory):
    traj = torch.load(trajectory)
    #print("traj shape=", traj.shape)
    colors = []
    lines = []
    max_context = 6
    max_pred = 8
    max_batch_ = len(traj)-1
    batch_size = 10

    colors = ['r', 'b', 'g', 'k', 'm', 'c','y'] # base colors 7


    ### EVENTS FLUSH with ax.scatter
    filenames = []
    batch_sequences_numpeds = [[8, 8, 9, 8, 8, 8, 8, 8, 8, 8]]
    plt.ion()
    fig, ax = plt.subplots()
    fig.suptitle("Ground truth", fontsize=30)
    init_traj = traj[0].cpu().numpy()
    print("batches =", len(traj)) #71
    print("init traj shape =", init_traj.shape) #(8, 81, 2)

    x, y = [], []
    x.append(init_traj[:max_pred,:max_context,0])
    y.append(init_traj[:max_pred,:max_context,1])

    #for ped in range(x[0].shape[1]):
    #    colors.append( [((np.random.uniform(0,255,1)/255)[0], (np.random.uniform(0,255, 1)/255)[0], (np.random.uniform(0,255,1)/255)[0])])

    for p in range(x[0].shape[1]):
        sc = ax.scatter(x[0][:,p], y[0][:,p], c=colors[p])
        lines.append(sc)
    plt.xlim(-2,15)
    plt.ylim(-2,15)

    plt.draw()
    for b in range(len(traj)-max_batch_):
        batch_numpeds = batch_sequences_numpeds[b]
        curr_total_peds = batch_numpeds[0]
        for seq in range(1, batch_size):
            seq_numpeds = batch_numpeds[seq]

            new_x = traj[b].cpu().numpy()[:max_pred,curr_total_peds:(curr_total_peds+max_context),0]
            new_y = traj[b].cpu().numpy()[:max_pred,curr_total_peds:(curr_total_peds+max_context),1]
            curr_total_peds += seq_numpeds

            # updating data values
            for p in range(new_x.shape[1]):
                lines[p].set_offsets(np.c_[new_x[:,p],new_y[:,p]])

            fig.canvas.draw_idle()
            # create file name and append it to a list
            filename = f'{b}-{seq}.png'
            filenames.append(filename)
        
            # save frame
            plt.savefig(filename)

            plt.pause(0.1)

    plt.ioff()

    # build gif
    with imageio.get_writer('zara01_gt_8ts_neurosym.gif', mode='I', fps =5, loop=3) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        
    # Remove files
    for filename in set(filenames):
        os.remove(filename)

    plt.waitforbuttonpress()



def main(args):
    path = args.traj_path
    plot_motion(path)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)



