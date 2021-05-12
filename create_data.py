import numpy as np
import argparse
import torch
from utils.utils import set_seed
import os
from config import load_data_config
from utils.create_cvs_data import create_cvs_data
from utils.create_pendulum_data import create_pendulum_data
from utils.create_double_pendulum_data import create_double_pendulum_data


def find_norm_params(data):
    mean = np.zeros(data.shape[2])
    std = np.zeros(data.shape[2])
    for feature in range(data.shape[2]):
        mean[feature] = data[:, :, feature].mean()
        std[feature] = data[:, :, feature].std()

    max_val = data.max()
    min_val = data.min()

    data_norm_params = {"mean": mean,
                        "std" : std,
                        "max" : max_val,
                        "min" : min_val}

    return data_norm_params


def add_noise(args, data):
    noisy_data = data + args.noise_std * np.random.normal(size=data.shape)
    return noisy_data


def create_mask(args, data_shape):
    revealed_n = int(round(args.mask_rate * args.seq_len))
    latent_mask = np.zeros(data_shape)
    max_val = int(args.seq_len * 0.75)
    for sample in range(data_shape[0]):
        train_latent_mask_ind = np.random.choice(range(max_val), size=revealed_n, replace=False)
        for mask_idx in train_latent_mask_ind:
            latent_mask[sample, mask_idx, :] = 1

    return latent_mask


def make_dataset(args):
    if not args.change_only_mask_rate:
        raw_data, latent_data, params_data = args.create_raw_data(args)

        buffer = int(round(raw_data.shape[0] * (1 - 0.1)))

        train_data = raw_data[:buffer]
        test_data = raw_data[buffer:]

        noisy_train_data = add_noise(args, train_data)
        noisy_test_data = add_noise(args, test_data)

        train_latent_data = latent_data[:buffer]
        test_latent_data = latent_data[buffer:]

        train_params_data = params_data[:buffer]
        train_params_data = {key: np.array([sample[key] for sample in train_params_data]) for key in train_params_data[0]}

        test_params_data = params_data[buffer:]
        test_params_data = {key: np.array([sample[key] for sample in test_params_data]) for key in test_params_data[0]}

        data_norm_params = find_norm_params(noisy_train_data)
        torch.save(train_params_data, args.save_path + 'train_params_data.pkl')
        torch.save(test_params_data, args.save_path + 'test_params_data.pkl')
        torch.save(train_latent_data, args.save_path + 'train_latent_data.pkl')
        torch.save(test_latent_data, args.save_path + 'test_latent_data.pkl')
        torch.save(data_norm_params, args.save_path + 'data_norm_params.pkl')
        torch.save(test_data, args.save_path + 'gt_test_data.pkl')

    else:
        train_latent_data = torch.load(args.save_path + 'train_latent_data.pkl')
        test_latent_data = torch.load(args.save_path + 'test_latent_data.pkl')

        dataset_dict = torch.load(args.save_path + 'processed_data.pkl')
        noisy_train_data = dataset_dict['train']
        noisy_test_data = dataset_dict['test']

    train_latent_mask = create_mask(args, train_latent_data.shape)
    test_latent_mask = create_mask(args, test_latent_data.shape)

    dataset_dict = {'train': noisy_train_data,
                    'test': noisy_test_data}

    grounding_data = {'train_latent': train_latent_data * train_latent_mask,
                      'train_latent_mask': train_latent_mask,
                      'test_latent': test_latent_data * test_latent_mask,
                      'test_latent_mask': test_latent_mask}
                
    unmasked_grounding_data = { 'train': train_latent_data,
                                'test': test_latent_data}

    unmasked_param_data = { 'train': train_params_data,
                            'test': test_params_data}

    torch.save(dataset_dict, args.save_path + 'processed_data.pkl')
    torch.save(grounding_data, args.save_path + 'grounding_data.pkl')
    torch.save(unmasked_grounding_data, args.save_path + 'unmasked_grounding_data.pkl')
    torch.save(unmasked_param_data, args.save_path + 'unmasked_param_data.pkl')

    args_dict = {'mask_rate': args.mask_rate, 'noise_std': args.noise_std, 'model': args.model}
    torch.save(args_dict, args.save_path + 'data_args.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--seq-len', type=int)
    parser.add_argument('--data-size', type=int)
    parser.add_argument('--delta-t', '-dt', type=float)
    parser.add_argument('--noise-std', type=float)
    parser.add_argument('--mask-rate', type=float, default=0.01)
    parser.add_argument('--model', choices=['cvs', 'pendulum', 'double_pendulum'], required=True)
    parser.add_argument('--change-only-mask-rate', action='store_true')
    parser.add_argument('--friction', action='store_true')
    parser.add_argument('--seed', type=int, default=12)

    parser.add_argument('--grounding-setup',  action='store_true')

    args = parser.parse_args()

    ## Get function
    if args.model == 'cvs':
        args.create_raw_data = create_cvs_data
    elif args.model == 'pendulum':
        args.create_raw_data = create_pendulum_data
    elif args.model == 'double_pendulum':
        args.create_raw_data = create_double_pendulum_data


    if args.grounding_setup:

        ########################################################
        ## Create pure dataset for artificial grounding

        pure_args = load_data_config(args)

        if args.model == 'pendulum':
            args.save_path = os.path.join(args.output_dir, 'pure_pendulum/') if not args.friction else os.path.join(args.output_dir, 'pure_pendulum_friction/')
        else:
            args.save_path = os.path.join(args.output_dir, 'pure_' + args.model + '/')

        args.seed = args.seed
        
        # Change size
        args.data_size = 50
        # Make it noise-less
        args.noise_std = 0.0


        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

        set_seed(args.seed)
        make_dataset(args)

        ################################################
        ## Create noisy dataset for traditionnal training
        noisy_args = load_data_config(args)

        if args.model == 'pendulum':
            args.save_path = os.path.join(args.output_dir, 'noisy_pendulum/') if not args.friction else os.path.join(args.output_dir, 'noisy_pendulum_friction/')
        else:
            args.save_path = os.path.join(args.output_dir, 'noisy_' + args.model + '/')
            
        # Change size
        args.data_size = 500
        # Make it noisy
        args.noise_std = 0.0002
        
        # Change the seed
        args.seed = args.seed + 1

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

        set_seed(args.seed)
        make_dataset(args)
    else:

        args = load_data_config(args)

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        set_seed(args.seed)
        make_dataset(args)
