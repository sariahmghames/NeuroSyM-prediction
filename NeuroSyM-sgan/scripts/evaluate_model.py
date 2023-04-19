import argparse
import os
import torch
import itertools
from attrdict import AttrDict
import numpy as np

from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path
from sgan.utils import int_tuple, bool_flag, get_total_norm

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)
parser.add_argument('--save', default=1, type=bool_flag)


def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm) # generator creation 
    generator.load_state_dict(checkpoint['g_state']) # we feed the generator , 
    generator.cuda()
    generator.train() # it calls forward()
    return generator


def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_



def evaluate_helper1(error, seq_start_end, st_dim):
    summ = []
    error = torch.stack(error, dim=st_dim)

    return error



def evaluate(args, loader, generator, num_samples):
    ade_outer, fde_outer, std_outer, fde_std_outer = [], [], [], []
    total_traj = 0
    traj_gt = []
    pred_traj = []
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch # gt for ground truth

            ade, fde, std = [], [], []
            total_traj += pred_traj_gt.size(1) 

            for _ in range(num_samples): 
                pred_traj_fake_rel = generator(
                    obs_traj, obs_traj_rel, seq_start_end
                ) 
                pred_traj_fake = relative_to_abs(
                    pred_traj_fake_rel, obs_traj[-1]
                )
                losses, loss_std = displacement_error(pred_traj_fake, pred_traj_gt, mode='raw')
                ade.append(losses)
                fde.append(final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'))
                std.append(loss_std)


            ade_sum = evaluate_helper(ade, seq_start_end) # over each seq_start_end in a batch
            fde_sum = evaluate_helper(fde, seq_start_end)
            std_sum = evaluate_helper1(std, seq_start_end, 2)
            fde_std_sum = evaluate_helper1(fde, seq_start_end, 1)

            ade_outer.append(ade_sum) # sum over 1 batch
            fde_outer.append(fde_sum)
            std_outer.append(std_sum)
            fde_std_outer.append(fde_std_sum)

            traj_gt.append(pred_traj_gt)
            pred_traj.append(pred_traj_fake)


        if (save):
            torch.save(traj_gt, 'zara01_test_gt.pt')
            torch.save(pred_traj, 'zara01_test_pred.pt')

        std_ade = torch.std(torch.cat(std_outer, dim=0), axis = 1)
        std_ade = torch.std(std_ade, axis=0)
        std_ade = torch.min(std_ade)
        fde_std_f = torch.std(torch.cat(fde_std_outer, dim=0), axis = 0)
        fde_std_f = torch.min(fde_std_f)

        ade = sum(ade_outer) / (total_traj * args.pred_len) # normalisation
        fde = sum(fde_outer) / (total_traj)
        #return ade, fde
        return ade, fde, std_ade, fde_std_f


def main(args):

    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]

    for path in paths:
        checkpoint = torch.load(path)
        generator = get_generator(checkpoint)
        _args = AttrDict(checkpoint['args'])
        print("loaded model args are :", _args)
        path = get_dset_path(_args.dataset_name, args.dset_type)
        _, loader = data_loader(_args, path)
        ade, fde, std, fde_std = evaluate(_args, args.save, loader, generator, args.num_samples)
        print('Dataset: {}, Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}, ADE-STD: {:.2f}, FDE-STD: {:.2f}'.format(
            _args.dataset_name, _args.pred_len, ade, fde, std, fde_std))

        #ade, fde= evaluate(_args, loader, generator, args.num_samples)
        #print('Dataset: {}, Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}'.format(
        #    _args.dataset_name, _args.pred_len, ade, fde))



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
