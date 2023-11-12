import argparse
import os
import torch

from attrdict import AttrDict

from sgan.data.loader_informed2_causal_v1 import data_loader
from sgan.models_informed2_causal_v1 import TrajectoryGenerator
from sgan.losses_informed_causal import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path
from sgan.utils import int_tuple, bool_flag, get_total_norm

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)
parser.add_argument('--save', default=1, type=bool_flag)
parser.add_argument('--delim_test', default='\t')


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
        batch_norm=args.batch_norm,
        embedding_dim_cvar1=args.embedding_dim_cvar1,
        embedding_dim_cvar2 = args.embedding_dim_cvar2,
        embedding_dim_cvar3 = args.embedding_dim_cvar3) # generator creation 
    generator.load_state_dict(checkpoint['g_state']) # we feed the generator , 
    generator.cuda()
    generator.train() # it calls forward()
    return generator


def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1) # stacking the num_samples runs along columns

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0) # dim 0 is the batch size
        _error = torch.min(_error) #min over the num_samples runs
        sum_ += _error # summing erros overs batch_size
    return sum_


def evaluate_helper2(error, seq_start_end):
    summ = []
    error = torch.stack(error, dim=1) # stacking the num_samples runs along columns

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        #print("_err dim =", _error.shape)
        _error = torch.std(_error, dim=0) # dim 0 is the batch size
        _error = torch.min(_error) #min over the num_samples runs
        summ.append(_error) # summing erros overs batch_size
    return torch.std(torch.tensor(summ))


def evaluate_helper3(error, seq_start_end):
    summ = []
    error = torch.stack(error, dim=1) # stacking the num_samples runs along columns

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        #print("_err dim =", _error.shape)
        _error = torch.mean(_error, dim=0) # dim 0 is the num_ped size in 1 example in a batch
        _error = torch.min(_error) #min over the num_samples runs
        summ.append(_error) # summing erros overs batch_size
    return torch.mean(torch.tensor(summ))

def evaluate_helper4(error, seq_start_end, st_dim):
    summ = []
    error = torch.stack(error, dim=st_dim)
    #error= torch.min(error, dim = 2)

    # for (start, end) in seq_start_end:
    #     start = start.item()
    #     end = end.item()
    #     _error = error[start:end]
    #     #_err = torch.std(_error, dim=0) 
    #     _er = torch.std(_error, axis=1) 
    #     _erro= torch.min(_er, axis = 1)
    #     #print("_erro shape =", _erro.shape)
    #     summ.append(_erro)
    # return summ
    return error


def evaluate_helper5(error, seq_start_end):
    summ = []
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _err = torch.std(_error, dim=0) 
        #_erro= torch.min(_er)
        summ.append(_err)
    return torch.tensor(summ)


def evaluate(args, save, loader, generator, num_samples):
    ade_outer, fde_outer, std_outer, fde_std_outer = [], [], [], []
    total_traj = 0
    traj_pred_gt = []
    traj_pred_est = []
    traj_obs = []
    batches = 0
    with torch.no_grad():
        for batch in loader:
            batches += 1
            batch_cut = [tensor.cuda() for tensor in batch[:-1]]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, traj_causal_inf_cvar1, traj_causal_inf_cvar2, traj_causal_inf_cvar3, seq_start_end) = batch_cut # gt for ground truth

            traj_weight = batch[-1]
            traj_causal_inf = [traj_causal_inf_cvar1, traj_causal_inf_cvar2, traj_causal_inf_cvar3]

            ade, fde, std = [], [], []
            total_traj += pred_traj_gt.size(1) # how many traj in 1 batch

            for _ in range(num_samples): # num_samples = 20 cz best_k = 20, then take the min in helper over the 20 runs
                pred_traj_fake_rel = generator(
                    obs_traj, obs_traj_rel, traj_weight, traj_causal_inf, seq_start_end
                ) # feeding the generator
                pred_traj_fake = relative_to_abs(
                    pred_traj_fake_rel, obs_traj[-1]
                )
                losses, loss_std = displacement_error(pred_traj_fake, pred_traj_gt, mode='raw')
                #losses = displacement_error(pred_traj_fake, pred_traj_gt, mode='raw')
                ade.append(losses)
                fde.append(final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'))
                #std.append(torch.std(loss_std, dim=1))
                std.append(loss_std)


            ade_sum = evaluate_helper(ade, seq_start_end) # over each seq_start_end in a batch
            fde_sum = evaluate_helper(fde, seq_start_end)
            #std_sum = evaluate_helper(std, seq_start_end)
            std_sum = evaluate_helper4(std, seq_start_end, 2)
            fde_std_sum = evaluate_helper4(fde, seq_start_end, 1)

            ade_outer.append(ade_sum) # sum over 1 batch
            fde_outer.append(fde_sum)
            std_outer.append(std_sum)
            fde_std_outer.append(fde_std_sum)


            traj_pred_gt.append(pred_traj_gt)
            traj_pred_est.append(pred_traj_fake)
            traj_obs.append(obs_traj)

        print("test batches = ", batches) # TO BE DONE: EXPORT THIS TO PLOT_MOTION_FULL
        if (save):
            torch.save(traj_pred_gt, 'THOR_full_test_pred_gt.pt')
            torch.save(traj_pred_est, 'THOR_full_test_pred_est.pt')
            torch.save(traj_obs, 'THOR_full_test_obs_gt.pt')
            #torch.save(peds_id_considered, 'THOR_full_test_peds_id.pt')


        #std_ade = sum(torch.tensor(std_outer)) / (total_traj) # same as torch.mean with helper3 case
        #std_outer = list(itertools.chain(*std_outer))
        #print(std_outer)
        #print("std shape = ",torch.tensor(std_outer).shape)
        std_ade = torch.std(torch.cat(std_outer, dim=0), axis = 1)
        std_ade = torch.std(std_ade, axis=0)
        std_ade = torch.min(std_ade)
        fde_std_f = torch.std(torch.cat(fde_std_outer, dim=0), axis = 0)
        fde_std_f = torch.min(fde_std_f)
        #std_ade = torch.std(torch.tensor(torch.stack(std_outer)), axis = 0)
        #std_ade = torch.mean(torch.tensor(std_outer))
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
        print("path=", path)
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
