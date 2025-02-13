"""
Author: Bruno Aristimunha
Training LDM with SleepEDFx or SHHS data.
Based on the tutorial from Monai Generative.

"""
import argparse

import torch
import torch.nn as nn

from generative.networks.nets import DiffusionModelUNet
from generative.networks.nets import AutoencoderKL
from generative.networks.schedulers import DDPMScheduler
from monai.config import print_config
from monai.utils import first, set_determinism
from omegaconf import OmegaConf
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter

from dataset.dataset import train_dataloader, valid_dataloader, get_trans
from models.ldm import UNetModel
from training import train_ldm
from util import log_mlflow, ParseListAction, setup_run_dir
from dataset.dataset_fnirs import (  # 替换原来的dataset导入
    fNIRSDataset,
    create_dataloaders
)
# print_config()
# for reproducibility purposes set a seed
import os
if os.path.exists('./project'):
    base_path = './project/'
    base_path_data = './data/'
else:
    # base_path = '/home/bru/PycharmProjects/DDPM-EEG/'  # original path
    base_path = os.getcwd()
    base_path_data = base_path
    
set_determinism(42)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        type=str,
        default="./config/config_ldm_fnirs.yaml",
        help="Path to config file with all the training parameters needed",
    )
    parser.add_argument(
        "--path_train_data",
        type=str,
        default='/home/jy/Documents/fnirs/treatment_response/generative_model_fnirs/diffusion_fNIRS/data/fnirs_MDD/train_data.npy',
        help="Path to fNIRS training data"
    )
    parser.add_argument(
        "--path_train_labels",
        type=str,
        default='/home/jy/Documents/fnirs/treatment_response/generative_model_fnirs/diffusion_fNIRS/data/fnirs_MDD/train_label.npy',
        help="Path to fNIRS training labels"
    )
    parser.add_argument(
        "--path_valid_data",
        type=str,
        default='/home/jy/Documents/fnirs/treatment_response/generative_model_fnirs/diffusion_fNIRS/data/fnirs_MDD/test_data.npy',
        help="Path to fNIRS validation data"
    )
    parser.add_argument(
        "--path_valid_labels",
        type=str,
        default='/home/jy/Documents/fnirs/treatment_response/generative_model_fnirs/diffusion_fNIRS/data/fnirs_MDD/test_label.npy',
        help="Path to fNIRS validation labels"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["fnirs", "edfx", "shhs", "shhsh"],
        default="fnirs"
    )
    parser.add_argument(
        "--num_channels",
        type=str, 
        action=ParseListAction,
        help="List of channel numbers for UNet architecture"
    )
    parser.add_argument(
        "--latent_channels",
        type=int,
        default=64,
        help="Number of latent channels from autoencoder"
    )
    parser.add_argument(
        "--spe",
        type=str,
        choices=["spectral", "no_spe"],
        default="no_spe",
        help="Whether to use spectral loss"
    )
    parser.add_argument(
        "--autoencoderkl_config_file_path",
        default="./config/config_aekl_eeg_fnirs.yaml",
        help="Path to autoencoder config file"
    )
    parser.add_argument(
        "--best_model_path",
        default="./outputs/fnirs/aekl_eeg_fnirs_no_spe_fnirs/best_model.pth",
        help="Path to pretrained autoencoder weights"
    )

    args = parser.parse_args()
    
    # Print all arguments
    print("\nArguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("\n")
    
    return args


class Stage1Wrapper(nn.Module):
    """Wrapper for stage 1 model as a workaround for
    the DataParallel usage in the training loop."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_mu, z_sigma = self.model.encode(x)
        z = self.model.sampling(z_mu, z_sigma)
        return z

def main(args):
    config = OmegaConf.load(args.config_file)

    set_determinism(seed=config.train.seed)
    print_config()

    run_dir, resume = setup_run_dir(config=config, args=args, base_path=base_path)

    # Getting write training and validation data

    writer_train = SummaryWriter(log_dir=str(run_dir / "train"))
    writer_val = SummaryWriter(log_dir=str(run_dir / "val"))
    trans = get_trans(args.dataset)
    # Getting data loaders
    # train_loader = train_dataloader(config=config, args=args)
    # val_loader = valid_dataloader(config=config, args=args)
    # 使用新的数据加载方式
    if args.dataset == "fnirs":
        data_paths = {
            'train_data': args.path_train_data,
            'train_labels': args.path_train_labels,
            'valid_data': args.path_valid_data,
            'valid_labels': args.path_valid_labels
        }
        train_loader, val_loader = create_dataloaders(
            data_paths,
            batch_size=config.train.batch_size,
            num_workers=config.train.num_workers
        )
    else:
        raise ValueError(f"Unsupported dataset")
    # Defining device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # Defining model
    config_aekl = OmegaConf.load(args.autoencoderkl_config_file_path)
    autoencoder_args = config_aekl.autoencoderkl.params

    # Remove any architecture overrides
    stage1 = AutoencoderKL(**autoencoder_args)

    state_dict = torch.load(args.best_model_path,
                            map_location=torch.device('cpu'))

    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:]  # remove `module.`
    #     new_state_dict[name] = v

    stage1.load_state_dict(state_dict)
    stage1.to(device)
    with torch.no_grad():
        with autocast(enabled=True):
            check_data = first(train_loader)['input_data'].to(device)
            print(f"\nInput data shape: {check_data.shape}")  # Should be [batch, 52, 376]
            z = stage1.encode_stage_2_inputs(check_data)
            print(f"Latent space shape: {z.shape}")  # Should be [batch, 64, 94]
            
            # Additional verification
            print("\nAutoencoder Configuration:")
            print(f"Input channels: {autoencoder_args['in_channels']}")
            print(f"Latent channels: {autoencoder_args['latent_channels']}")
            print(f"Num channels: {autoencoder_args['num_channels']}")

    autoencoderkl = Stage1Wrapper(model=stage1)

    #########################################################################
    # Diffusion model part
    # spatial_dims: 1
    # in_channels: 1
    # out_channels: 1
    # num_channels: [1, 2, 4]
    # latent_channels: 1
    # num_res_blocks: 2
    # norm_num_groups: 1
    # attention_levels: [false, false, false]
    # with_encoder_nonlocal_attn: false
    # with_decoder_nonlocal_attn: false
    #
    # diffusion = DiffusionModelUNet(
    #     spatial_dims=1,
    #     in_channels=1,
    #     out_channels=1,
    #     num_res_blocks=[8,4],
    #     num_channels=[1,2],
    #     attention_levels=(False, False),
    #     norm_num_groups=1,
    #     norm_eps=1e-6,
    #     resblock_updown=False,
    #     num_head_channels=1,
    #     with_conditioning=False,
    #     transformer_num_layers=1,
    #     cross_attention_dim=None,
    #     num_class_embeds=None,
    #     upcast_attention=False,
    #     use_flash_attention=False,
    # )
    #print(diffusion)
    parameters = config['model']['params']['unet_config']['params']
    parameters.update({
        'image_size': 94,
        'in_channels': 64,
        'out_channels': 64
    })

    diffusion = UNetModel(
        image_size=94,
        in_channels=64,
        out_channels=64,
        model_channels=128,
        attention_resolutions=[],
        num_res_blocks=1,
        channel_mult=[1, 2],  # No downsampling
        num_heads=1,
        use_scale_shift_norm=True,
        conv_resample=False,
        kernel_size=3,
        padding=1
    )

    if torch.cuda.device_count() > 1:
        autoencoderkl = torch.nn.DataParallel(autoencoderkl)
        diffusion = torch.nn.DataParallel(diffusion)

    autoencoderkl.eval()
    
    autoencoderkl = autoencoderkl.to(device)
    diffusion.to(device)

    scheduler = DDPMScheduler(num_train_timesteps=1000,
                              beta_start=0.0015, beta_end=0.0195)
    
    scheduler.to(device)
    print(f"Scaling factor set to {1 / torch.std(z)}")
    scale_factor = 1 / torch.std(z)

    #inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

    optimizer = torch.optim.Adam(diffusion.parameters(), lr=1e-4)

    best_loss = float("inf")
    start_epoch = 0

    print(f"Starting Training")
    val_loss = train_ldm(
        model=diffusion,
        stage1=autoencoderkl,
        scheduler=scheduler,
        start_epoch=start_epoch,
        best_loss=best_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        n_epochs=config.train.n_epochs,
        eval_freq=config.train.eval_freq,
        writer_train=writer_train,
        writer_val=writer_val,
        device=device,
        run_dir=run_dir,
        scale_factor=scale_factor,
    )

    log_mlflow(
        model=diffusion,
        config=config,
        args=args,
        run_dir=run_dir,
        val_loss=val_loss,
    )

    print("\nDiffusion UNet Configuration:")
    print(f"UNet image_size: {parameters['image_size']}")
    print(f"UNet in_channels: {parameters['in_channels']}")
    print(f"UNet channel_mult: {parameters['channel_mult']}")
    print(f"UNET kernel_size: {parameters.get('kernel_size', 'default')}")
    print(f"UNET padding: {parameters.get('padding', 'default')}")


if __name__ == "__main__":
    args = parse_args()
    main(args)


##
#
#python src/train_autoencoderkl.py --dataset edfx
#
#python src/train_ldm.py --dataset edfx