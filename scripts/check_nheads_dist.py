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
        # patch_size,
        device=device,
    ).to(device)
    num_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print("Number of parameters: ", num_params)
    print("Aspect ratio: ", d_model / num_layers)
    print("Number of heads: ", num_heads)

    return num_params, num_heads


def check_nheads_dist():
    yml_files = glob.glob("configs/nheads_scaling/*.yml")
    N_list = []
    nheads_list = []

    for config_file in yml_files:
        print(config_file)
        N, nh = train(config_file)
        N_list.append(N)
        nheads_list.append(nh)

    plt.scatter(N_list, nheads_list)
    plt.xlabel("Number of parameters")
    plt.ylabel("Number of heads")
    plt.xlim(1e4, 1e8)
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig("plots/nheads_dist.pdf", bbox_inches="tight")


if __name__ == "__main__":
    check_nheads_dist()
