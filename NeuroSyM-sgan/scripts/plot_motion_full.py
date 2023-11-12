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
parser.add_argument('--traj_path_pred_gt', type=str)
parser.add_argument('--traj_path_pred_est', type=str)
parser.add_argument('--traj_path_obs', type=str)
parser.add_argument('--path_peds_id', type=str)




def plot_motion(trajectory_pred_gt, trajectory_pred_est, trajectory_obs, peds_id):

	traj_pred_gt = torch.load(trajectory_pred_gt)
	traj_pred_est = torch.load(trajectory_pred_est)
	traj_obs = torch.load(trajectory_obs)
	peds_id = np.load(peds_id, allow_pickle = True)
	batch_size = 10

	print("traj_obs=", len(traj_obs))
	print("traj_pred_gt=", len(traj_pred_gt))
	print("traj_pred_est=", len(traj_pred_est))
	print("peds_id=", len(peds_id))
	num_seq = traj_obs[0].shape # 8, 60, 2 
	print("num_seq in first batch= ", traj_pred_gt[0].shape)
	#print("traj obs= ", traj_obs[0])

	# TO IMPORT
	num_peds_per_seq_in_exp_1_run_1 = [7, 7, 7, 7, 6, 6, 5, 5, 5, 5, 5, 5, 5, 4, 3, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 4, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 5, 5, 3, 3, 5, 5, 5, 5, 4, 4, 3, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 5, 6, 6, 6, 6, 6, 4, 4, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 7, 6, 6, 5, 5, 5, 5, 4, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 4, 4, 4, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 4, 4, 4, 3, 3, 3, 3, 3, 4, 4, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 7, 7, 7, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]

	peds_pred_traj_est = {}
	peds_pred_traj_gt = {}
	peds_obs_traj = {}

	for batch in range(len(traj_obs)):
		num_peds_in_batch = num_peds_per_seq_in_exp_1_run_1[batch*batch_size:(batch*batch_size)+batch_size]
		peds_ids_in_batch = peds_id[batch*batch_size:(batch*batch_size)+batch_size]
		for seq_ind in range(len(peds_ids_in_batch)):
			one_seq_ids = peds_ids_in_batch[seq_ind]
			for ped_ind in range(len(one_seq_ids)):
				ped = one_seq_ids[ped_ind]
				one_seq_len = len(one_seq_ids)
				if ped in peds_obs_traj.keys():
					peds_obs_traj[ped]["x"] = traj_obs[batch][:, int(ped_ind+((batch_size*batch)+np.sum(num_peds_in_batch[:seq_ind])))  ,0]
					peds_obs_traj[ped]["y"] = traj_obs[batch][:, int(ped_ind+((batch_size*batch)+np.sum(num_peds_in_batch[:seq_ind])))  ,1]
					peds_pred_traj_gt[ped]["x"] = traj_pred_gt[batch][:, int(ped_ind+((batch_size*batch)+np.sum(num_peds_in_batch[:seq_ind]))), 0]
					peds_pred_traj_gt[ped]["y"] = traj_pred_gt[batch][:, int(ped_ind+((batch_size*batch)+np.sum(num_peds_in_batch[:seq_ind]))) ,1]
					peds_pred_traj_est[ped]["x"] = traj_pred_est[batch][:, int(ped_ind+((batch_size*batch)+np.sum(num_peds_in_batch[:seq_ind]))) ,0]
					peds_pred_traj_est[ped]["y"] = traj_pred_est[batch][:, int(ped_ind+((batch_size*batch)+np.sum(num_peds_in_batch[:seq_ind]))) ,1]
				else:
					peds_obs_traj[ped] = {"x":[], "y": []}
					peds_pred_traj_gt[ped] = {"x":[], "y": []}
					peds_pred_traj_est[ped] = {"x":[], "y": []}
					print("ped_ind =", ped_ind)
					print("cumulative num peds =", num_peds_in_batch[:seq_ind])
					print("####################", ped_ind+((batch_size*batch)+np.sum(num_peds_in_batch[:seq_ind])))
					peds_obs_traj[ped]["x"] = traj_obs[batch][:, int(ped_ind+((batch_size*batch)+np.sum(num_peds_in_batch[:seq_ind]))) ,0]
					peds_obs_traj[ped]["y"] = traj_obs[batch][:, int(ped_ind+((batch_size*batch)+np.sum(num_peds_in_batch[:seq_ind]))) ,1]
					peds_pred_traj_gt[ped]["x"] = traj_pred_gt[batch][:, int(ped_ind+((batch_size*batch)+np.sum(num_peds_in_batch[:seq_ind]))) ,0]
					peds_pred_traj_gt[ped]["y"] = traj_pred_gt[batch][:, int(ped_ind+((batch_size*batch)+np.sum(num_peds_in_batch[:seq_ind]))) ,1]
					peds_pred_traj_est[ped]["x"] = traj_pred_est[batch][:, int(ped_ind+((batch_size*batch)+np.sum(num_peds_in_batch[:seq_ind]))) ,0]
					peds_pred_traj_est[ped]["y"] = traj_pred_est[batch][:, int(ped_ind+((batch_size*batch)+np.sum(num_peds_in_batch[:seq_ind]))) ,1]



    # batches_to_plot = 1
    # for i in range(batches_to_plot):
    #     #if (i == 0):
    #     #    continue
    #     traj_pred_gt = torch.load(trajectory_pred_gt)[i]
    #     traj_pred_est = torch.load(trajectory_pred_est)[i]
    #     traj_obs = torch.load(trajectory_obs)[i]
    #     print("traj_obs=", traj_obs.shape)
    #     #print("traj_pred_gt=", traj_pred_gt.shape)
    #     num_seq = traj_obs.dim()
    #     if (num_seq ==3):
    #         traj = torch.cat((traj_obs, traj_pred_est), axis=0) # axis is 1 if there are 10 sequences in a batch
    #     else:
    #         traj = torch.cat((traj_obs, traj_pred_est), axis=1) # axis is 1 if there are 10 sequences in a batch
    #     print("traj shape=", traj.shape)
    #     colors = []
    #     lines = []

    #     max_pred = 8
    #     max_batch_ = len(trajectory_obs)-1
    #     pred_len = traj_pred_gt.size(0)
    #     obs_len = traj_obs.size(0)

    #     print("obs_len=", obs_len)


    #     colors = ['b', 'g', 'r', 'k', 'm', 'c','y'] # base colors 7



    #     ### EVENTS FLUSH with ax.scatter
    #     filenames = []

    #     #batch_sequences_numpeds = [[8, 8, 9, 8, 8, 8, 8, 8, 8, 8]] # batch of 10 sequences, each seq has 8, 8, 9, ... num_peds, in order, so it sums-up to 81 - UCY dataset
    #     batch_sequences_numpeds = [[3, 3, 3, 3, 3, 3, 3, 3, 2, 2]] # ARENA batch 2
    #     #batch_sequences_numpeds = [[4, 4, 4, 4, 4, 4, 3, 2, 2, 2]] # ARENA dataset test batch 1, TO BE DONE: IMPORT THIS FROM EVALUATE.py AND TRAJECTORIES.py - ARENA dataset
    #     #batch_sequences_numpeds = [[7, 7, 7, 7, 6, 6, 5, 5, 5, 5]] # THOR dataset test1
    #     batch_size = len(batch_sequences_numpeds[0])
    #     batch_size = batch_size - 4 # 4 if batch 1 and 2 if batch 2
    #     max_context = np.min(batch_sequences_numpeds[0][:batch_size])



        
    #     # if (num_seq ==3):
    #     #     batch_sequences_numpeds = traj.shape[1]
    #     # else:
    #     #     for seq in range(traj.shape[0]):
    #     #         batch_sequences_numpeds.append(traj[seq].shape[2])

    #     plt.ion()
    #     fig, ax = plt.subplots()
    #     #fig.suptitle("ARENA Causal NeuroSym SGAN", fontsize=20)
    #     fig.suptitle("ARENA Causal NeuroSym SGAN EST", fontsize=20)
    #     init_traj = traj.cpu().numpy() # traj[0] is first batch
    #     print("batches =", len(traj)) #71
    #     print("init traj shape =", init_traj.shape) #(8, 81, 2)
    #     #print("batch_1 shape =", traj[1].cpu().numpy().shape) #(8, 39, 2)
    #     #print("batch_2 shape =", traj[2].cpu().numpy().shape) #(8, 48, 2)
    #     #print("batch_1 shape =", traj[5].cpu().numpy().shape) #(8, 31, 2)
    #     #print("batch_2 shape =", traj[10].cpu().numpy().shape) #(8, 41, 2)

    #     x, y = [], []
    #     x.append(init_traj[:(obs_len+pred_len),:max_context,0])
    #     y.append(init_traj[:(obs_len+pred_len),:max_context,1])

    #     #for ped in range(x[0].shape[1]):
    #     #    colors.append( [((np.random.uniform(0,255,1)/255)[0], (np.random.uniform(0,255, 1)/255)[0], (np.random.uniform(0,255,1)/255)[0])])

    #     for p in range(x[0].shape[1]):
    #         sco = ax.scatter(x[0][:obs_len,p], y[0][:obs_len,p], c=colors[p], marker = 'o', s=50) # x[0] is 1st sample/sequence in first batch
    #         scp = ax.scatter(x[0][obs_len:obs_len+pred_len,p], y[0][obs_len:obs_len+pred_len,p], c=colors[p], marker = '*', s=100)
    #         lines.append([sco, scp])

    #     ## ARENA
    #     #plt.xlim(-14,-6)
    #     #plt.ylim(3,10)

    #     #plt.xlim(-15,-8)
    #     #plt.ylim(0,10)

    #     plt.draw()
    #     if (len(batch_sequences_numpeds[batches_to_plot-1])>1): # we have more than 1 seq in 1 batch
    #         for b in range(len(trajectory_obs)-max_batch_):
    #             batch_numpeds = batch_sequences_numpeds[b]
    #             curr_total_peds = batch_numpeds[0]
    #             for seq in range(1, batch_size):
    #                 seq_numpeds = batch_numpeds[seq]

    #                 new_x = traj.cpu().numpy()[:(obs_len+pred_len),curr_total_peds:(curr_total_peds+max_context),0]
    #                 new_y = traj.cpu().numpy()[:(obs_len+pred_len),curr_total_peds:(curr_total_peds+max_context),1]
    #                 curr_total_peds += seq_numpeds

    #                 # updating data values
    #                 for p in range(new_x.shape[1]):
    #                     lines[p][0].set_offsets(np.c_[new_x[:obs_len,p],new_y[:obs_len,p]])
    #                     lines[p][1].set_offsets(np.c_[new_x[obs_len:obs_len+pred_len,p],new_y[obs_len:obs_len+pred_len,p]])

    #                 fig.canvas.draw_idle()
    #                 # create file name and append it to a list
    #                 filename = f'{b}-{seq}.png'
    #                 filenames.append(filename)
                
    #                 # save frame
    #                 plt.savefig(filename)

    #                 plt.pause(0.1)

    #         plt.ioff()

    #         # build gif
    #         with imageio.get_writer('ARENA_8ts_causal_neurosym_est.gif', mode='I', fps =10, loop=20) as writer:
    #             for filename in filenames:
    #                 image = imageio.imread(filename)
    #                 writer.append_data(image)
                
    #         # Remove files
    #         for filename in set(filenames):
    #             os.remove(filename)

    #         plt.waitforbuttonpress()



def main(args):
    path_pred_gt = args.traj_path_pred_gt
    path_pred_est = args.traj_path_pred_est
    path_obs = args.traj_path_obs
    path_peds_id = args.path_peds_id
    plot_motion(path_pred_gt, path_pred_est, path_obs, path_peds_id)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)



