import os
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
import yaml
import argparse
import json
import wandb
import os
import sys

cwd = os.getcwd()
sys.path.insert(0, cwd + "/../timetransformers")

from data_handling import TimeSeriesDataset, load_datasets
from utils import GradualWarmupScheduler
import Transformer


def train(config):
    with open(config, "r") as file:
        config = yaml.safe_load(file)

    # Accessing the configuration values
    train_split = config["train"]["train_split"]
    max_seq_length = config["train"]["max_seq_length"]
    batch_size = config["train"]["batch_size"]
    test_batch_size = config["train"]["test_batch_size"]
    total_training_steps = config["train"]["total_training_steps"]
    early_stopping = config["train"]["early_stopping"]
    warmup_steps = config["train"]["warmup_steps"]
    evaluation_interval = config["train"]["evaluation_interval"]
    test_size = config["train"]["test_size"]
    max_learning_rate = config["train"]["max_LR"]

    # Transformer parameters
    output_dim = config["transformer"]["output_dim"]
    d_model = config["transformer"]["d_model"]
    num_heads = config["transformer"]["num_heads"]
    num_layers = config["transformer"]["num_layers"]
    d_ff = config["transformer"]["d_ff"]
    dropout = config["transformer"]["dropout"]
    num_distribution_layers = config["transformer"]["num_distribution_layers"]
    loss_function = config["transformer"]["loss_func"]

    # Datasets
    datasets_to_load = config["datasets"]

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        # else "mps" if torch.backends.mps.is_available() else "cpu"
        else "cpu"
    )
    print(f"Using {device}")
    if device.type == "cuda":
        num_workers = 11
    elif device.type == "mps" or device.type == "cpu":
        num_workers = 6

    print("Number of workers: ", num_workers)

    # First lets download the data and make a data loader
    print("Loading data...")
    (
        training_data_list,
        test_data_list,
        train_masks,
        test_masks,
    ) = load_datasets(datasets_to_load, train_split)

    train_dataset = TimeSeriesDataset(training_data_list, max_seq_length, train_masks)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    test_dataset = TimeSeriesDataset(
        test_data_list,
        max_seq_length,
        test_masks,
        test=True,
        test_size=test_size,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    print("Training dataset size: ", train_dataset.__len__())
    print("Test dataset size: ", test_dataset.__len__())

    print("Total number of training tokens:", train_dataset.total_length())
    print("Total number of test tokens:", test_dataset.total_length())

    print("Train batches: ", len(train_dataloader))
    print("Test batches: ", len(test_dataloader))

    # Now lets make a transformer

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

    # Now lets train it!
    # max_learning_rate = 1e-4
    # print(f"Max learning rate: {max_learning_rate}")
    optimizer = optim.AdamW(transformer.parameters(), lr=max_learning_rate)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_training_steps, eta_min=1e-6
    )
    scheduler = GradualWarmupScheduler(
        optimizer,
        total_warmup_steps=warmup_steps,
        after_scheduler=cosine_scheduler,
    )

    config["Nparams"] = num_params

    wandb.init(
        project="timetransformers",
        entity="timetransformers",
        name=f"maxLRscaling_{num_params}_{max_learning_rate}",
        config=config,
    )
    file_name = f"results/maxLRscaling_{num_params}_{max_learning_rate}.json"
    model_file_name = f"results/maxLRscaling_{num_params}_{max_learning_rate}_final.pt"

    train_steps = []
    train_losses = []
    test_steps = []
    test_losses = []
    MSE_test_losses = []
    CRPS_test_losses = []

    min_loss = 1e10
    patience_counter = 0
    step_counter = 0
    should_break = (
        False  # This will be used to break the loops should the train loss diverge
    )

    # Initialize tqdm progress bar
    pbar = tqdm(total=total_training_steps, desc="Training", position=0)

    while step_counter < total_training_steps:
        for batch in train_dataloader:
            transformer.train()
            if step_counter >= total_training_steps:
                break

            train, true, mask = batch

            batched_data = train.to(device)
            batched_data_true = true.to(device)
            optimizer.zero_grad()
            output = transformer(batched_data, custom_mask=mask.to(device))

            if loss_function == "Gaussian":
                loss = transformer.Gaussian_loss(output, batched_data_true)
            elif loss_function == "studentT":
                loss = transformer.studentT_loss(
                    output, batched_data_true, mask=mask.to(device)
                )
            elif loss_function == "MSE":
                loss = transformer.MSE(output, batched_data_true)

            train_losses.append(loss.item())
            train_steps.append(step_counter)

            # Update tqdm bar with each step
            wandb.log({"train_loss": loss.item(), "step": step_counter})
            pbar.set_description(f"Step {step_counter}: Loss {loss.item():.5f}")
            pbar.update(1)

            loss.backward()
            optimizer.step()
            scheduler.step()

            if len(train_losses) > 1000 and (
                abs(train_losses[-1] - train_losses[-2]) > 1.5
            ):
                print("Derivative of loss is too large, breaking")
                should_break = True
                break

            if len(train_losses) > 1000 and (train_losses[0] < train_losses[-1]):
                print("Loss is larger than initial loss, breaking")
                should_break = True
                break

            if len(train_losses) > 1000 and (
                torch.sum(torch.isnan(torch.tensor(train_losses))) > 0
            ):
                print("Loss is NaN, breaking")
                should_break = True
                break

            if step_counter % evaluation_interval == 0:
                transformer.eval()
                total_test_loss = 0
                total_MSE_test_loss = 0
                total_CRPS_test_loss = 0
                total_test_samples = 0

                with torch.no_grad():  # Disable gradient calculation
                    for batch in test_dataloader:
                        train, true, mask = batch
                        current_batch_size = train.shape[0]
                        batched_data = train.to(device)
                        batched_data_true = true.to(device)
                        output = transformer(batched_data)

                        if loss_function == "Gaussian":
                            test_loss = transformer.Gaussian_loss(
                                output, batched_data_true
                            )
                            # test_loss_MSE = transformer.MSE(output, batched_data_true)
                            test_loss_CRPS = transformer.crps_student_t_approx(
                                output, batched_data_true
                            )
                            # total_MSE_test_loss += (
                            #     test_loss_MSE.item() * current_batch_size
                            # )
                            total_CRPS_test_loss += (
                                test_loss_CRPS.item() * current_batch_size
                            )

                        if loss_function == "studentT":
                            test_loss = transformer.studentT_loss(
                                output, batched_data_true, mask=mask.to(device)
                            )
                            test_loss_MSE = transformer.MSE(
                                output, batched_data_true, mask=mask.to(device)
                            )
                            test_loss_CRPS = transformer.crps_student_t_approx(
                                output, batched_data_true, mask=mask.to(device)
                            )
                            total_MSE_test_loss += (
                                test_loss_MSE.item() * current_batch_size
                            )
                            total_CRPS_test_loss += (
                                test_loss_CRPS.item() * current_batch_size
                            )

                        elif loss_function == "MSE":
                            test_loss = transformer.MSE(
                                output, batched_data_true, mask=mask.to(device)
                            )

                        total_test_loss += test_loss.item() * current_batch_size
                        total_test_samples += current_batch_size

                average_test_loss = total_test_loss / total_test_samples
                if loss_function == "Gaussian" or loss_function == "studentT":
                    average_MSE_test_loss = total_MSE_test_loss / total_test_samples
                    average_CRPS_test_loss = total_CRPS_test_loss / total_test_samples
                    wandb.log(
                        {"CRPS_test_loss": average_CRPS_test_loss, "step": step_counter}
                    )
                    wandb.log(
                        {"MSE_test_loss": average_MSE_test_loss, "step": step_counter}
                    )

                MSE_test_losses.append(average_MSE_test_loss)
                CRPS_test_losses.append(average_CRPS_test_loss)
                test_losses.append(average_test_loss)
                test_steps.append(step_counter)
                wandb.log({"test_loss": average_test_loss, "step": step_counter})

                if average_test_loss < min_loss:
                    torch.save(
                        transformer.state_dict(),
                        f"results/maxLRscaling_{num_params}_{max_learning_rate}_best.pt",
                    )

                # Early stopping
                if average_test_loss < min_loss:
                    min_loss = average_test_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Finally, save the data up to this point
                train_info = {
                    "train_losses": train_losses,
                    "train_epochs": train_steps,
                    "test_losses": test_losses,
                    "CRPS_test_losses": CRPS_test_losses,
                    "MSE_test_losses": MSE_test_losses,
                    "test_epochs": test_steps,
                    "model_file_name": f"{model_file_name}",
                }
                # Writing data to a JSON file
                full_train_info = config | train_info
                with open(file_name, "w") as file:
                    json.dump(full_train_info, file, indent=4)

            step_counter += 1

        if patience_counter > early_stopping:
            print("Early stopping")
            break

        if should_break:
            break

    pbar.close()

    config["diverge"] = should_break

    # Finally, lets save the losses
    if loss_function == "Gaussian" or loss_function == "studentT":
        train_info = {
            "train_losses": train_losses,
            "train_epochs": train_steps,
            "test_losses": test_losses,
            "CRPS_test_losses": CRPS_test_losses,
            "MSE_test_losses": MSE_test_losses,
            "test_epochs": test_steps,
            "model_file_name": f"{model_file_name}",
        }
    else:
        train_info = {
            "train_losses": train_losses,
            "train_epochs": train_steps,
            "test_losses": test_losses,
            "test_epochs": test_steps,
            "model_file_name": f"{model_file_name}",
        }
    print(f"Saving final model weights to {model_file_name}")
    torch.save(
        transformer.state_dict(),
        f"{model_file_name}",
    )

    # Writing data to a JSON file
    full_train_info = config | train_info
    with open(file_name, "w") as file:
        json.dump(full_train_info, file, indent=4)

    wandb.finish()

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script configuration.")
    parser.add_argument(
        "config_file", type=str, help="Path to the configuration YAML file."
    )
    args = parser.parse_args()

    train(args.config_file)
