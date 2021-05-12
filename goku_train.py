import argparse
import numpy as np
import torch
from utils import ODE_dataset, utils
import models
import os
from config import load_goku_train_config

######  REMOVE THIS 
import matplotlib.pyplot as plt

def plot_samples(pred_x, mini_batch, file='sample.png'):

    plt.figure() # specifying the overall grid size
    if len(list(pred_x.shape)) == 4:
        for i in range(10):
            plt.subplot(2,10,i+1) 
            plt.imshow(pred_x.cpu().detach().numpy()[0,5*i,:,:])
            plt.subplot(2,10,i+11) 
            plt.imshow(mini_batch.cpu().detach().numpy()[0,5*i,:,:])
    elif len(list(pred_x.shape)) == 3:
        plt.subplot(2,3,1) 
        plt.plot(pred_x.cpu().detach().numpy()[0,:,0])
        plt.subplot(2,3,2) 
        plt.plot(pred_x.cpu().detach().numpy()[0,:,1])
        plt.subplot(2,3,3) 
        plt.plot(pred_x.cpu().detach().numpy()[0,:,2])
        plt.subplot(2,3,4) 
        plt.plot(mini_batch.cpu().detach().numpy()[0,:,0])
        plt.subplot(2,3,5) 
        plt.plot(mini_batch.cpu().detach().numpy()[0,:,1])
        plt.subplot(2,3,6) 
        plt.plot(mini_batch.cpu().detach().numpy()[0,:,2])
    
    plt.savefig(file)
    plt.close()

def plot_latent(pred_z, mini_batch):
    plt.figure() # specifying the overall grid size

    plt.subplot(2,1,1) 
    plt.plot(pred_z.cpu().detach().numpy()[0,:,0])
    plt.subplot(2,1,2) 
    plt.plot(mini_batch.cpu().detach().numpy()[0,:,0])
    
    plt.savefig("latent.png")
    plt.close()

def validate_goku(args, model, val_dataloader, device):
    model.eval()
    with torch.no_grad():
        for val_batch in val_dataloader:
            val_batch = val_batch.to(device)
            t_arr = torch.arange(0.0, end=args.seq_len * args.delta_t, step=args.delta_t, device=device)
            predicted_batch, _, _, _, _, _, _, _, _ = model(val_batch, t=t_arr, variational=False)
            val_loss = ((predicted_batch - val_batch)**2).mean((0, 1)).sum()

            # plot_samples(predicted_batch, val_batch, 'val_sample.png')

    model.train()
    return val_loss

def compute_losses(args, epoch, i_batch, n_batch, model, t, input_batch, latent_batch, param_batch, device):

    # Forward step
    pred_x, pred_z, pred_params, z_0, z_0_loc, z_0_log_var, params, params_loc, params_log_var = model(
        input_batch, t=t, variational=True)

    # plot_samples(pred_x, input_batch)


    ## Calculate losses:
    # Parameter loss
    param_loss = 0
    for k in list(pred_params.keys()):
        param_loss += args.grounding_loss * ((param_batch[k].to(device) - pred_params[k])**2).mean()

    # Reconstruction loss
    rec_loss = ((pred_x - input_batch) ** 2).mean((0, 1)).sum()

    # Initiation loss
    init_loss = args.grounding_loss * ((latent_batch[:,0,:] - pred_z[:,0,:])**2).mean(0).sum()

    # KL loss
    kl_annealing_factor = utils.annealing_factor_sched(args.kl_start_af, args.kl_end_af,
                                                    args.kl_annealing_epochs, epoch, i_batch,
                                                    n_batch)
    analytic_kl_z0 = utils.normal_kl(z_0_loc, z_0_log_var,
                                    torch.zeros_like(z_0_loc),
                                    torch.zeros_like(z_0_log_var)).sum(1).mean(0)
    analytic_kl_params = utils.normal_kl(params_loc, params_log_var,
                                        torch.zeros_like(params_loc),
                                        torch.zeros_like(params_log_var)).sum(1).mean(0)
    kl_loss = kl_annealing_factor * (analytic_kl_z0 + analytic_kl_params)

    return param_loss, rec_loss, init_loss, kl_loss


def train(args):
    # General settings
    utils.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if not args.cpu else torch.device('cpu')

    # Define data transform
    data_transforms = utils.create_transforms(args)

    # Create model - see models/GOKU.py for options
    model = models.__dict__["create_goku_" + args.model](ode_method=args.method).to(device)
    print('Model: GOKU - %s created with %d parameters.' % (args.model, sum(p.numel() for p in model.parameters())))


    ########################################
    ## Perform artificial grounding steps ##
    ########################################
    if args.grounding:
        # Get training data and make dataloaders
        ds_ground_latent_train = ODE_dataset.ODEDataSet(file_path=args.grounding_data_path + 'unmasked_grounding_data.pkl',
                                        ds_type='train',
                                        seq_len=args.seq_len,
                                        random_start=False)
        ds_ground_input_train = ODE_dataset.ODEDataSet(file_path=args.grounding_data_path + 'processed_data.pkl',
                                        ds_type='train',
                                        seq_len=args.seq_len,
                                        random_start=False,
                                        transforms=data_transforms)
        ds_ground_param_train = ODE_dataset.ParamDataSet(file_path=args.grounding_data_path + 'unmasked_param_data.pkl',
                                        ds_type='train')

        ground_latent_train_dataloader = torch.utils.data.DataLoader(ds_ground_latent_train, batch_size=args.mini_batch_size)
        ground_input_train_dataloader = torch.utils.data.DataLoader(ds_ground_input_train, batch_size=args.mini_batch_size)
        ground_param_train_dataloader = torch.utils.data.DataLoader(ds_ground_param_train, batch_size=args.mini_batch_size)

        # Get validation data and make dataloaders
        ds_ground_latent_val = ODE_dataset.ODEDataSet(file_path=args.grounding_data_path + 'unmasked_grounding_data.pkl',
                                        ds_type='val',
                                        seq_len=args.seq_len,
                                        random_start=False)
        ds_ground_input_val = ODE_dataset.ODEDataSet(file_path=args.grounding_data_path + 'processed_data.pkl',
                                        ds_type='val',
                                        seq_len=args.seq_len,
                                        random_start=False,
                                        transforms=data_transforms)
        ds_ground_param_val = ODE_dataset.ParamDataSet(file_path=args.grounding_data_path + 'unmasked_param_data.pkl',
                                        ds_type='val')

        ground_latent_val_dataloader = torch.utils.data.DataLoader(ds_ground_latent_val, batch_size=len(ds_ground_latent_val))
        ground_input_val_dataloader = torch.utils.data.DataLoader(ds_ground_input_val, batch_size=len(ds_ground_input_val))
        ground_param_val_dataloader = torch.utils.data.DataLoader(ds_ground_param_val, batch_size=len(ds_ground_param_val))

        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        # Saving container
        grounding_dat_train = {'loss': [],
                         'epoch': [],
                         'param_loss': [],
                         'init_loss': [],
                         'reconstruction_loss':[],
                         'kl_loss': []
                         }
        grounding_dat_val = {'loss': [],
                         'epoch': [],
                         'param_loss': [],
                         'init_loss': [],
                         'reconstruction_loss':[],
                         'kl_loss': []
                         }

        for epoch in range(args.grounding_epoch):
            
            grounding_dat_train_epoch = {'loss': [],
                                   'param_loss': [],
                                   'init_loss': [],
                                   'reconstruction_loss':[],
                                   'kl_loss': [] }
            grounding_dat_val_epoch = {'loss': [],
                                   'param_loss': [],
                                   'init_loss': [],
                                   'reconstruction_loss':[],
                                   'kl_loss': [] }

            for i_batch, (input_batch, latent_batch, param_batch) in enumerate(zip(ground_input_train_dataloader, ground_latent_train_dataloader, ground_param_train_dataloader)):

                input_batch = input_batch.to(device)
                latent_batch = latent_batch.to(device)

                ######################
                ## Get training losses
                t = torch.arange(0.0, end=args.seq_len * args.delta_t, step=args.delta_t, device=device)
                param_loss, rec_loss, init_loss, kl_loss = compute_losses(args, epoch, i_batch, len(ground_input_train_dataloader), model, t, input_batch, latent_batch, param_batch, device)

                ## Print info
                print("    Batch %d" % (i_batch))
                grounding_dat_train_epoch['param_loss'].append(param_loss.item())
                print("        Param Loss = %.4f" % (grounding_dat_train_epoch['param_loss'][-1]))
                grounding_dat_train_epoch['reconstruction_loss'].append(rec_loss.item())
                print("        Reconstruction Loss = %.4f" % (grounding_dat_train_epoch['reconstruction_loss'][-1]))
                grounding_dat_train_epoch['init_loss'].append(init_loss.item())
                print("        Init Loss = %.4f" % (grounding_dat_train_epoch['init_loss'][-1]))
                grounding_dat_train_epoch['kl_loss'] .append(kl_loss.item())
                print("        KL Loss = %.4e" % (grounding_dat_train_epoch['kl_loss'][-1]))

                # Total loss
                loss = param_loss + rec_loss + init_loss + kl_loss
                grounding_dat_train_epoch['loss'] .append(loss.item())

                # Backward step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            ######################
            ## Get validation losses
            model.eval()
            with torch.no_grad():

                input_val_batch, latent_val_batch, param_val_batch = next(iter(zip(ground_input_val_dataloader, ground_latent_val_dataloader, ground_param_val_dataloader)))
                
                input_val_batch = input_val_batch.to(device)
                latent_val_batch = latent_val_batch.to(device)

                param_loss, rec_loss, init_loss, kl_loss = compute_losses(args, epoch, 1, 1, model, t, input_val_batch, latent_val_batch, param_val_batch, device)
            
                print("    Validation batch")
                grounding_dat_val_epoch['param_loss'].append(param_loss.item())
                print("        Param Loss = %.4f" % (grounding_dat_val_epoch['param_loss'][-1]))
                grounding_dat_val_epoch['reconstruction_loss'].append(rec_loss.item())
                print("        Reconstruction Loss = %.4f" % (grounding_dat_val_epoch['reconstruction_loss'][-1]))
                grounding_dat_val_epoch['init_loss'].append(init_loss.item())
                print("        Init Loss = %.4f" % (grounding_dat_val_epoch['init_loss'][-1]))
                grounding_dat_val_epoch['kl_loss'] .append(kl_loss.item())
                print("        KL Loss = %.4e" % (grounding_dat_val_epoch['kl_loss'][-1]))
                grounding_dat_val_epoch['loss'] .append((param_loss + rec_loss + init_loss + kl_loss).item())
            model.train()

            # Mean train statistic over past epoch
            grounding_dat_train['epoch'].append(epoch)
            for k, v in grounding_dat_train_epoch.items():
                grounding_dat_train_epoch[k] = np.mean(v)
                grounding_dat_train[k].append(grounding_dat_train_epoch[k])
            # Mean val statistic over past epoch
            grounding_dat_val['epoch'].append(epoch)
            for k, v in grounding_dat_val_epoch.items():
                grounding_dat_val_epoch[k] = np.mean(v)
                grounding_dat_val[k].append(grounding_dat_val_epoch[k])

            print("[*Grounding Epoch* %d/%d]  Train Loss = %.4f" % (epoch + 1, args.grounding_epoch,  grounding_dat_train_epoch['loss']))
            print("                          Validation Loss = %.4f" % (grounding_dat_val_epoch['loss']))

        try:
            os.mkdir(args.save_path)
        except OSError as error:
            print("Path already exists")

        torch.save(grounding_dat_train, os.path.join(args.save_path, 'grounding_data_train.pt'))
        torch.save(grounding_dat_val, os.path.join(args.save_path, 'grounding_data_val.pt'))

    ###############################################3
    ## True Training ##
    ##########################

    #################################
    # Create train and test datasets:

    # Get training data and make dataloaders
    ds_latent_train = ODE_dataset.ODEDataSet(file_path=args.data_path + 'unmasked_grounding_data.pkl',
                                    ds_type='train',
                                    seq_len=args.seq_len,
                                    random_start=False)
    ds_input_train = ODE_dataset.ODEDataSet(file_path=args.data_path + 'processed_data.pkl',
                                    ds_type='train',
                                    seq_len=args.seq_len,
                                    random_start=False,
                                    transforms=data_transforms)
    ds_param_train = ODE_dataset.ParamDataSet(file_path=args.data_path + 'unmasked_param_data.pkl',
                                    ds_type='train')

    latent_train_dataloader = torch.utils.data.DataLoader(ds_latent_train, batch_size=args.mini_batch_size)
    input_train_dataloader = torch.utils.data.DataLoader(ds_input_train, batch_size=args.mini_batch_size)
    param_train_dataloader = torch.utils.data.DataLoader(ds_param_train, batch_size=args.mini_batch_size)

    # Get validation data and make dataloaders
    ds_latent_val = ODE_dataset.ODEDataSet(file_path=args.data_path + 'unmasked_grounding_data.pkl',
                                    ds_type='val',
                                    seq_len=args.seq_len,
                                    random_start=False)
    ds_input_val = ODE_dataset.ODEDataSet(file_path=args.data_path + 'processed_data.pkl',
                                    ds_type='val',
                                    seq_len=args.seq_len,
                                    random_start=False,
                                    transforms=data_transforms)
    ds_param_val = ODE_dataset.ParamDataSet(file_path=args.data_path + 'unmasked_param_data.pkl',
                                    ds_type='val')

    latent_val_dataloader = torch.utils.data.DataLoader(ds_latent_val, batch_size=len(ds_latent_val))
    input_val_dataloader = torch.utils.data.DataLoader(ds_input_val, batch_size=len(ds_input_val))
    param_val_dataloader = torch.utils.data.DataLoader(ds_param_val, batch_size=len(ds_param_val))

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # L1 error on validation set (not test set!) for early stopping
    best_model = models.__dict__["create_goku_" + args.model](ode_method=args.method).to(device)
    best_val_loss = np.inf

    # Saving container
    dat_train = {'loss': [],
                 'epoch': [],
                 'param_loss': [],
                 'init_loss': [],
                 'reconstruction_loss':[],
                 'kl_loss': []
                 }
    dat_val = {'loss': [],
               'epoch': [],
               'param_loss': [],
               'init_loss': [],
               'reconstruction_loss':[],
               'kl_loss': []
               }
    ## True training
    for epoch in range(args.num_epochs):
            
        dat_train_epoch = {'loss': [],
                           'param_loss': [],
                           'init_loss': [],
                           'reconstruction_loss':[],
                           'kl_loss': [] }
        dat_val_epoch = {'loss': [],
                         'param_loss': [],
                         'init_loss': [],
                         'reconstruction_loss':[],
                         'kl_loss': [] }

        for i_batch, (input_batch, latent_batch, param_batch) in enumerate(zip(input_train_dataloader, latent_train_dataloader, param_train_dataloader)):
            
            input_batch = input_batch.to(device)
            latent_batch = latent_batch.to(device)

            # Forward step
            t = torch.arange(0.0, end=args.seq_len * args.delta_t, step=args.delta_t, device=device)
            pred_x, pred_z, pred_params, z_0, z_0_loc, z_0_log_var, params, params_loc, params_log_var = model(
                input_batch, t=t, variational=True)

            param_loss, rec_loss, init_loss, kl_loss = compute_losses(args, epoch, i_batch, len(input_train_dataloader), model, t, input_batch, latent_batch, param_batch, device)

            ## Print info
            print("    Batch %d" % (i_batch))
            dat_train_epoch['param_loss'].append(param_loss.item())
            print("        Param Loss = %.4f" % (dat_train_epoch['param_loss'][-1]))
            dat_train_epoch['reconstruction_loss'].append(rec_loss.item())
            print("        Reconstruction Loss = %.4f" % (dat_train_epoch['reconstruction_loss'][-1]))
            dat_train_epoch['init_loss'].append(init_loss.item())
            print("        Init Loss = %.4f" % (dat_train_epoch['init_loss'][-1]))
            dat_train_epoch['kl_loss'] .append(kl_loss.item())
            print("        KL Loss = %.4e" % (dat_train_epoch['kl_loss'][-1]))

            # Total loss
            loss = rec_loss + kl_loss
            dat_train_epoch['loss'] .append(loss.item())

            # Backward step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ######################
        ## Get validation losses
        model.eval()
        with torch.no_grad():

            input_val_batch, latent_val_batch, param_val_batch = next(iter(zip(input_val_dataloader, latent_val_dataloader, param_val_dataloader)))
            
            input_val_batch = input_val_batch.to(device)
            latent_val_batch = latent_val_batch.to(device)

            param_loss, rec_loss, init_loss, kl_loss = compute_losses(args, epoch, 1, 1, model, t, input_val_batch, latent_val_batch, param_val_batch, device)
        
            print("    Validation batch")
            dat_val_epoch['param_loss'].append(param_loss.item())
            print("        Param Loss = %.4f" % (dat_val_epoch['param_loss'][-1]))
            dat_val_epoch['reconstruction_loss'].append(rec_loss.item())
            print("        Reconstruction Loss = %.4f" % (dat_val_epoch['reconstruction_loss'][-1]))
            dat_val_epoch['init_loss'].append(init_loss.item())
            print("        Init Loss = %.4f" % (dat_val_epoch['init_loss'][-1]))
            dat_val_epoch['kl_loss'] .append(kl_loss.item())
            print("        KL Loss = %.4e" % (dat_val_epoch['kl_loss'][-1]))
            dat_val_epoch['loss'] .append((rec_loss + kl_loss).item())
        model.train()

        # Mean train statistic over past epoch
        dat_train['epoch'].append(epoch)
        for k, v in dat_train_epoch.items():
            dat_train_epoch[k] = np.mean(v)
            dat_train[k].append(dat_train_epoch[k])
        # Mean val statistic over past epoch
        dat_val['epoch'].append(epoch)
        for k, v in dat_val_epoch.items():
            dat_val_epoch[k] = np.mean(v)
            dat_val[k].append(dat_val_epoch[k])

        print("[Epoch %d/%d]  Train Loss = %.4f" % (epoch + 1, args.num_epochs,  dat_train_epoch['loss']))
        print("                 Validation Loss = %.4f" % (dat_val_epoch['loss']))
    
    try:
        os.mkdir(args.save_path)
    except OSError as error:
        print("Path already exists")

    torch.save(dat_train, os.path.join(args.save_path, 'data_train.pt'))
    torch.save(dat_val, os.path.join(args.save_path, 'data_val.pt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    # Run parameters
    parser.add_argument('-n', '--num-epochs', type=int)
    parser.add_argument('-mbs', '--mini-batch-size', type=int)
    parser.add_argument('--seed', type=int, default=1)

    # Data parameters
    parser.add_argument('-sl', '--seq-len', type=int)
    parser.add_argument('--delta-t', type=float)
    parser.add_argument('--data-path', type=str)
    parser.add_argument('--norm', type=str, choices=['zscore', 'zero_to_one'], default=None)

    # Optimizer parameters
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.001)
    parser.add_argument('-wd', '--weight-decay', type=float, default=0.0)

    # Model parameters
    parser.add_argument('-m', '--method', type=str, default='rk4')
    parser.add_argument('--model', type=str, choices=['pendulum', 'double_pendulum', 'cvs', 'pendulum_friction'],
                        required=True)

    # KL Annealing factor parameters
    parser.add_argument('--kl-annealing-epochs', type=int)
    parser.add_argument('--kl-start-af', type=float)
    parser.add_argument('--kl-end-af', type=float)

    # Saving arguments
    parser.add_argument('--run-id', type=str, default='some_run')
    parser.add_argument('--save-path', type=str, default='./runs/')
    parser.add_argument('--checkpoints-dir', type=str, default='checkpoints/')
    parser.add_argument('--cpu', action='store_true')

    # Artificial Grounding
    parser.add_argument('--grounding', action='store_true')
    parser.add_argument('--grounding-epoch', type=int, default=10)

    args = parser.parse_args()
    args = load_goku_train_config(args)
    args.save_path = os.path.join(args.save_path, args.model, args.run_id)
    args.checkpoints_dir = args.save_path

    if args.model == 'pendulum':
        args.grounding_data_path = os.path.join(args.data_path, 'pure_pendulum/')
        args.data_path = os.path.join(args.data_path, 'noisy_pendulum/')
    if args.model == 'double_pendulum':
        args.grounding_data_path = os.path.join(args.data_path, 'pure_double_pendulum/')
        args.data_path = os.path.join(args.data_path, 'noisy_double_pendulum/')
    elif args.model == 'cvs':
        args.grounding_data_path = os.path.join(args.data_path, 'pure_cvs/')
        args.data_path = os.path.join(args.data_path, 'noisy_cvs/')

    if not os.path.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)

    train(args)
