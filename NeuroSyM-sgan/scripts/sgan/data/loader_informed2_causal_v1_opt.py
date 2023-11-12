from torch.utils.data import DataLoader

from sgan.data.trajectories_informed2_causal_v1_opt import TrajectoryDataset, seq_collate


def data_loader(args, path):
    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim,
        labels_dir = args.labels_dir,
        filename = args.cndfilename,
        cinf_dir = args.cinf_dir,
        cfilename = args.cfilename,
        div_data = args.div_data) # dset is instance of the class initialised

    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.loader_num_workers,
        collate_fn=seq_collate)
    return dset, loader
