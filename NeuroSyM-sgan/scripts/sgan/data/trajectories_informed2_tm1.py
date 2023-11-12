import logging
import os
import math
import itertools
import numpy as np
import torch
import glob
import json
import csv
from datetime import date

from torch.utils.data import Dataset
from sgan.data.qtc import qtcc1

logger = logging.getLogger(__name__)


#with open('config.json', 'r') as f:
#    config = json.load(f)

def seq_collate(data):
	(obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list, non_linear_ped_list, loss_mask_list,  obs_traj_weight) = zip(*data)

	_len = [len(seq) for seq in obs_seq_list]
	cum_start_idx = [0] + np.cumsum(_len).tolist()
	seq_start_end = [[start, end]
	                 for start, end in zip(cum_start_idx, cum_start_idx[1:])]

	# Data format: batch, input_size, seq_len
	# LSTM input format: seq_len, batch, input_size
	obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
	pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
	obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
	pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
	non_linear_ped = torch.cat(non_linear_ped_list)
	loss_mask = torch.cat(loss_mask_list, dim=0)
	seq_start_end = torch.LongTensor(seq_start_end)
	#traj_causal_inf_cvar1 = torch.cat(traj_causal_inf_cvar1, dim=0).permute(2, 0, 1)
	#traj_causal_inf_cvar2 = torch.cat(traj_causal_inf_cvar2, dim=0).permute(2, 0, 1)
	#traj_causal_inf_cvar3 = torch.cat(traj_causal_inf_cvar3, dim=0).permute(2, 0, 1)
	out = [
	    obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped,
	    loss_mask, seq_start_end, obs_traj_weight]
    
	return tuple(out)


def read_file(_path, delim='\t'):
	data = []
	if delim == 'tab':
	    delim = '\t'
	elif delim == 'space':
	    delim = ' '
	with open(_path, 'r') as f:
		for line in f:
			line = line.strip().split(delim)
			line = [float(i) for i in line]
			data.append(line)
	return np.asarray(data)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


class TrajectoryDataset(Dataset): # A custom Dataset class must implement three functions: __init__, __len__, and __getitem__.
	"""Dataloder for the Trajectory datasets"""
	def __init__(
	    self, data_dir, obs_len=8, pred_len=12, skip=1, threshold=0.002,
	    min_ped=1, delim='\t', labels_dir = os.getcwd(), filename= "cnd_labels.txt", cinf_dir = os.getcwd(), cfilename = "causal_vars.json", div_data = 1):
		"""
		Args:
		- data_dir: Directory containing dataset files in the format
		<frame_id> <ped_id> <x> <y>
		- obs_len: Number of time-steps in input trajectories
		- pred_len: Number of time-steps in output trajectories
		- skip: Number of frames to skip while making the dataset
		- threshold: Minimum error to be considered for non linear traj
		when using a linear predictor
		- min_ped: Minimum number of pedestrians that should be in a seqeunce
		- delim: Delimiter in the dataset files
		"""
		super(TrajectoryDataset, self).__init__()

		self.data_dir = data_dir
		self.obs_len = obs_len
		self.pred_len = pred_len
		self.skip = skip
		self.seq_len = self.obs_len + self.pred_len
		self.delim = delim
		self.repeat = 2
		self.labels_dir = labels_dir
		self.cndfilename = filename
		# TO BE CONTINUED
		self.cinf_dir = cinf_dir
		self.cfilename = cfilename


		all_files = os.listdir(self.data_dir)
		all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
		num_peds_in_seq = []
		peds_id_in_seq = []
		frames_in_seq = []
		seq_list = [] #  each row is 1 sequence and each sequence is len(seq_len)=nb_frames/seq and in each frame there is nb of peds considered then each x and y 
		seq_list_weight = []
		seq_list_causal_inf = {}
		seq_list_rel = []
		loss_mask_list = []
		non_linear_ped = []

		causal_labels = self.causalabel() 

		for l in range(1, causal_labels.size(0)):
			seq_list_causal_inf["curr_seq_causal_inf_cvar"+str(l)] = []

		for path in all_files:
			print("csv path=", path)
			data = read_file(path, delim)
			#print("#################data =", data[:16])
			frames = np.unique(data[:, 0]).tolist()
			frame_data = []
			for frame in frames:
				frame_data.append(data[data[:, 0] == frame, :])

			num_sequences = int(
			   math.ceil((len(frames) - self.seq_len + 1) / skip))

			total_sequences = 0
			print("num_sequences=", num_sequences) # 20548
			#last_data = np.concatenate(
			#	    frame_data[20547:20547+ self.seq_len], axis=0)
			#print("last_data=", last_data)
			#debug_data = np.concatenate(frame_data[11878:11878+ self.seq_len], axis=0)
			#print("debug_data=", debug_data)
			for idx in range(0, num_sequences * self.skip +1, skip):
				curr_seq_data = np.concatenate(
				    frame_data[idx:idx + self.seq_len], axis=0)

				#print("##############curr_seq_data=", curr_seq_data)

				peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
				#if (idx == 0):
				# 	print("curr seq data=", curr_seq_data)
				# 	print("peds in curr seq=", peds_in_curr_seq)
				curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
				                         self.seq_len))
				curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
				curr_seq_qtc_AB = []
				curr_seq_ped_ids = []
				#curr_seq_weight = np.zeros((math.factorial(len(peds_in_curr_seq))/len(peds_in_curr_seq), 2, self.seq_len)) # weight are copied (same) along dim = 1, 

				curr_loss_mask = np.zeros((len(peds_in_curr_seq),
				                           self.seq_len))
				num_peds_considered = 0
				peds_id_considered = []
				frames_peds_considered = []
				_non_linear_ped = []
				for _, ped_id in enumerate(peds_in_curr_seq): # _ is the index in the list and ped_in is corresponding value

					curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
					curr_ped_seq = np.around(curr_ped_seq, decimals=4)
					# Check if the seq len is right
					pad_front = frames.index(curr_ped_seq[0, 0]) - idx # index of first frame where this ped_id appears
					pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1 # index of last frame where this ped_id appears
					# if (idx == 0):
					# 	print("pad front=", pad_front)
					# 	print("pad end=", pad_end)
					# 	print("pad diff=", pad_end-pad_front)
					# if (idx ==0):
					# 	print("curr_ped_seq len =================", curr_ped_seq.shape[0])
					# 	print("curr_ped_seq =================", curr_ped_seq)
					if ((pad_end - pad_front != self.seq_len) or (curr_ped_seq.shape[0]!= self.seq_len)):
						continue

					curr_ped_seq = np.transpose((curr_ped_seq[:, 2:4])/div_data) # (2: are x and y) so final shape = 2 x seq len
					curr_ped_seq = curr_ped_seq # shape : 2 x seq len

					# Make coordinates relative ("with respect to frames")
					rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
					rel_curr_ped_seq[:, 1:] = \
					    curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1] # current - prev
					_idx = num_peds_considered
					# if (curr_ped_seq.shape[1]!= 16): # bug in ped_num = 8 at sequence 11879, 
					# 	print("idx =", idx) # bug at idx = 11878
					# 	print("curr_ped_seq=", curr_ped_seq)

					curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
					curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
					# Linear vs Non-Linear Trajectory
					_non_linear_ped.append(
					    poly_fit(curr_ped_seq, pred_len, threshold))
					curr_loss_mask[_idx, pad_front:pad_end] = 1
					curr_seq_ped_ids.append(ped_id)
					num_peds_considered += 1 # the num of peds that appear in all the frames considered for the seq len (so in all the 16 frames)
					peds_id_considered.append(int(ped_id))
					frames_peds_considered.append(curr_seq_data[curr_seq_data[:, 1] == ped_id,0])
                
				ind_excl = [(x*num_peds_considered)+x for x in range(num_peds_considered)]
				idx_qtc_weight = 0
				#idx_qtc_weight_prev = 0
				# TO BE CONTINUED
				idx_causal_weight = 0
				#curr_seq_causal_inf = {}


				peds_comb_list = list(itertools.product(curr_seq[:num_peds_considered], repeat=self.repeat))
				peds_comb_list_causal = peds_comb_list[:num_peds_considered]

				#for x in itertools.product(curr_seq[:num_peds_considered], curr_seq[1:num_peds_considered]):
				if (self.repeat <= num_peds_considered and num_peds_considered!=1):
					total_sequences +=1
					#print("curr_seq_ped_ids =", curr_seq_ped_ids) # TO BE DONE: EXPORT THIS TO PLOT_MOTION_FULL
					#curr_seq_weight = np.ones((int(math.factorial(num_peds_considered)/math.factorial(num_peds_considered-self.repeat)), 1, self.seq_len-1)) # weight are copied (same) along dim = 1, 
					curr_seq_weight = torch.ones((num_peds_considered * num_peds_considered, 2, 2)) # weight are copied (same) along dim = 1, 
					# TO BE CONTINUED
					#curr_seq_causal = np.ones((num_peds_considered, 2, 2)) # weight are copied (same) along dim = 1, 
					#curr_seq_causal = []
					
					#for l in range(1, (causal_labels.size(0))):
					#	curr_seq_causal_inf["curr_seq_causal_inf_cvar"+str(l)] = curr_seq_causal

					for tup_idx, tup in enumerate(peds_comb_list):
						if tup_idx not in ind_excl:
							curr_seq_qtc_AB = qtcc1(tup[0][:, :2].transpose(), tup[1][:, :2].transpose(), qbits = 4)
							labels = [self.labelme(list(x)) for x in curr_seq_qtc_AB]
							curr_seq_weight[idx_qtc_weight, 0, -1] = torch.tensor([float(i) for i in labels])
							curr_seq_weight[idx_qtc_weight, 1, -1] = torch.tensor([float(i) for i in labels])
							#curr_seq_weight[idx_qtc_weight, 0, :1] = torch.tensor([[0]+[float(i) for i in labels]])
							#curr_seq_weight[idx_qtc_weight, 1, :1] = torch.tensor([[0]+[float(i) for i in labels]])
							idx_qtc_weight += 1
						else:
							idx_qtc_weight += 1


					# for tup_idx, tup in enumerate(peds_comb_list_causal):
					# 	# TO BE CONTINUED
					# 	for key, value in curr_seq_causal_inf.items():
					# 		#print("curr##############", curr_seq_causal_inf[key].shape)
					# 		l = int(key[len(key)-1])
					# 		curr_seq_causal_inf[key][idx_causal_weight, 0, :] = np.array([causal_labels[0,l],0])
					# 		curr_seq_causal_inf[key][idx_causal_weight, 1, :] = np.array([causal_labels[0,l],0])
					# 	idx_causal_weight += 1


				else:
				    curr_seq_weight = torch.zeros((1 , 2, 1))
				    #curr_seq_weight[:, :, 0] = 0
				    # TO BE CONTINUED
				    #for key, value in curr_seq_causal_inf.items():
				    #	curr_seq_causal_inf[key] = np.ones((1 , 2, 2))
				    #	curr_seq_causal_inf[key][:, :, -1] = 0
				
				#print("_non_linear_peds in seq=", _non_linear_ped)

				if num_peds_considered > min_ped:
					#if (idx ==0):
					#	print("num_peds_considered = ", num_peds_considered)
					#print("seq_list=", seq_list)
					non_linear_ped += _non_linear_ped # (appending as extension of elements) trajectories of peds in 1 seq to another seq
					#print("non_linear_peds in seq=", non_linear_ped)
					num_peds_in_seq.append(num_peds_considered)
					peds_id_in_seq.append(np.asarray(peds_id_considered))
					frames_in_seq.append(np.asarray(frames_peds_considered))
					loss_mask_list.append(curr_loss_mask[:num_peds_considered])
					seq_list.append(curr_seq[:num_peds_considered])
					seq_list_weight.append(curr_seq_weight)
					# TO BE CONTINUED
					#for key, value in curr_seq_causal_inf.items():
					#	seq_list_causal_inf[key].append(value) ######################################## ADJUST HERE FOR EACH CVAR ALONE
					seq_list_rel.append(curr_seq_rel[:num_peds_considered])

				##if (idx in range(10)):
				##	print("curr_seq_ped_ids =", curr_seq_ped_ids) # TO BE DONE: EXPORT THIS TO PLOT_MOTION_FULL
			#print("num_peds_in_seq =", num_peds_in_seq) # TO BE DONE: EXPORT THIS TO PLOT_MOTION_FULL
			print("total_sequences=", total_sequences) # TO BE DONE: EXPORT THIS TO PLOT_MOTION_FULL

		self.num_seq = len(seq_list)
		seq_list = np.concatenate(seq_list, axis=0) # num_sequences, nb peds , seq_len, 2 # num sequences are batched
		
		#seq_list_weight = np.concatenate(seq_list_weight, axis =0) # num_sequences, nb peds , seq_len, 2 # num sequences are batched


		seq_list_rel = np.concatenate(seq_list_rel, axis=0)
		loss_mask_list = np.concatenate(loss_mask_list, axis=0)
		non_linear_ped = np.asarray(non_linear_ped)
		#peds_id_in_seq = np.asarray(peds_id_in_seq)
		#print("################################", peds_id_in_seq)
		np.save('THOR_full_test_peds_id', peds_id_in_seq )
		np.save('THOR_full_test_frames', frames_in_seq )

		# Convert numpy -> Torch Tensor
		self.obs_traj = torch.from_numpy(seq_list[:, :, :self.obs_len]).type(torch.float) # shape : num_sequences, nb peds , seq_len, (2?) 
		self.obs_traj_weight = seq_list_weight # shape : num_sequences, nb peds , seq_len, (2?) ; 
		
		# TO BE CONTINUED
		# traj_cinf = []
		# for key, value in seq_list_causal_inf.items():
		# 	traj_cinf.append(torch.from_numpy(np.concatenate(value, axis=0)).type(torch.float))
		# 	#traj_cinf.append([torch.from_numpy(item).type(torch.float) for item in value])

		# self.traj_causal_inf_cvar1 = traj_cinf[0]
		# self.traj_causal_inf_cvar2 = traj_cinf[1]
		# self.traj_causal_inf_cvar3 = traj_cinf[2]

		self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len:]).type(torch.float)
		#self.pred_traj_weight = torch.from_numpy(seq_list_weight[:][:, :, self.obs_len:]).type(torch.float)

		self.obs_traj_rel = torch.from_numpy(
		    seq_list_rel[:, :, :self.obs_len]).type(torch.float)
		self.pred_traj_rel = torch.from_numpy(
		    seq_list_rel[:, :, self.obs_len:]).type(torch.float)
		self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
		self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)

		cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist() # np.cumsum of num of peds considered over all sequences
		#print("cum_start_idx=", cum_start_idx) # [0, 3, 6, 9, ... 1712, 1714]
		self.seq_start_end = [
		    (start, end)
		    for start, end in zip(cum_start_idx, cum_start_idx[1:]) # end is the nb of peds considered in 1 seq .. [(0,3), (3, 6), (6,9) ....] the first is (0,3) because the first appended sequence has 3 peds considered so 3 trajectories of len = seq_len. so start is index of 1st appended traj in 1 sequence and end is index of last traj considered (or last pedestrian) in that sequence. we are not appending the ped_id considered in this sequence
		]
		#self.peds_id_in_seq = torch.from_numpy(peds_id_in_seq)
		#print("seq_start_end=", self.seq_start_end[:10])

	def __len__(self): # The __len__ function returns the number of samples in our dataset.
	    return self.num_seq


	def labelme(self, curr_seq_qtc_AB):
		curr_dir = os.getcwd()
		txt_file =  curr_dir + "/" + self.labels_dir + self.cndfilename
		qtc = []
		with open(txt_file) as f:
		    labels = [line.strip() for line in f.readlines()] # removes newline \n character at end of each effective line 
		    for x in labels:
		        qtc.append(x.split(' ')[1])
		    #print("qtcAB=",qtc)
		    qtcAB_idx = qtc.index(str(curr_seq_qtc_AB).replace(' ', '')) 
		    qtcAB_label = labels[qtcAB_idx].split()[0]

		return qtcAB_label


	# TO BE CONTINUED 
	def causalabel1(self, ):
		curr_dir = os.getcwd()
		json_file =  curr_dir + "/" + self.cinf_dir + self.cfilename
		with open(json_file) as f:
			causal_vars = [f["cvar1"], f["cvar2"], f["cvar3"], f["cvar4"] ]
			causal_nb = len(causal_vars)
			causal_inf = np.ones((causal_vars, causal_vars))
		return causal_vars, causal_inf


	def causalabel(self, ):
		curr_dir = os.getcwd()
		csv_file =  curr_dir + "/" + self.cinf_dir + self.cfilename

		with open(csv_file) as f:
			csv_reader = csv.reader(f, delimiter=',')
			causal_nb = np.sum(1 for row in csv_reader) 
			causal_inf = []
			for row in csv_reader:
				print("row=", row)
				causal_inf.append(row)
			#causal_inf = torch.cat(causal_inf, axis = 0)
			causal_inf = torch.zeros((causal_nb, causal_nb))
			causal_inf[0,1] = 0.0861
			causal_inf[0,2] = 0.0773
			causal_inf[0,3] = 0.0588
		return causal_inf


	def __getitem__(self, index): # The __getitem__ function loads and returns a sample from the dataset at the given index idx. called internally when we create an instance of the class and pass an index to it
		# called by DataLoader of torch when batching, 

		start, end = self.seq_start_end[index]
		# TO BE CONTINUED
		out = [
		    self.obs_traj[start:end, :], self.pred_traj[start:end, :], # self.obs_traj[start:end, :] shape is torch.Size([5,2,8]) , torch.Size([5,2,8]), torch.Size([8,2,8]) ...
		    self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
		    self.non_linear_ped[start:end], self.loss_mask[start:end, :],  self.obs_traj_weight[index]]

		return out
