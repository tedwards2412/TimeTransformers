import os
import torch
import yaml
import glob

# This is just until temporary implementation
import os
import sys
import matplotlib.pyplot as plt

cwd = os.getcwd()
sys.path.insert(0, cwd + "/../timetransformers")

import Transformer


def train(config):
    with open(config, "r") as file:
        config = yaml.safe_load(file)
    # Transformer parameters
    output_dim = config["transformer"]["output_dim"]
    d_model = config["transformer"]["d_model"]
    num_heads = config["transformer"]["num_heads"]
    num_layers = config["transformer"]["num_layers"]
    d_ff = config["transformer"]["d_ff"]
    dropout = config["transformer"]["dropout"]
    num_distribution_layers = config["transformer"]["num_distribution_layers"]
    max_seq_length = config["train"]["max_seq_length"]
    max_LR = config["train"]["max_LR"]

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        # else "mps" if torch.backends.mps.is_available() else "cpu"
        else "cpu"
    )
    print(f"Using {device}")

    transformer = Transformer.Decoder_Transformer(
        output_dim,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_length,
        dropout,
        num_distribution_layers,
        device=device,
    ).to(device)
    num_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print("Number of parameters: ", num_params)
    print("Aspect ratio: ", d_model / num_layers)

    return num_params, max_LR


def check_LR_dist():
    yml_files = glob.glob("configs/max_LR_scaling/*.yml")
    N_list = []
    LR_list = []

    for config_file in yml_files:
        N, LR = train(config_file)
        N_list.append(N)
        LR_list.append(LR)

    print(N_list, LR_list)
    plt.scatter(N_list, LR_list)
    plt.xlabel("Number of parameters")
    plt.ylabel("Maximum Learning Rate")
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig("plots/LR_dist.pdf", bbox_inches="tight")


if __name__ == "__main__":
    check_LR_dist()
