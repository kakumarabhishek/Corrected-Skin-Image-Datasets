from __future__ import print_function, division
import torch
from torchvision import transforms, models
import pandas as pd
import numpy as np
import os
import warnings
import torch.nn as nn
import torch.optim as optim
import time
import copy
from comet_ml import Experiment

import argparse
import random
from accelerate import Accelerator
from utils import get_partitions, SkinDataset, flatten

warnings.filterwarnings("ignore")

accelerator = Accelerator(
    # https://huggingface.co/docs/accelerate/v0.26.1/en/package_reference/state#accelerate.state.AcceleratorState
    mixed_precision="fp16",  # Set to "no" if your GPU does not support FP16 operations.
    rng_types=["torch", "cuda", "generator"],
)

experiment = Experiment(project_name="INSERT_EXPERIMENT_NAME_FOR_COMET")


def train_model(
    label, dataloaders, device, dataset_sizes, model, criterion, optimizer, num_epochs=2
):
    since = time.time()
    training_results = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 0.0
    for epoch in range(num_epochs):
        print("-" * 10)
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data.
            for batch in dataloaders[phase]:
                inputs = batch["image"].to(device)
                labels = batch[label].long().to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    inputs = inputs.float()
                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        with accelerator.autocast():
                            accelerator.backward(loss)
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            training_results.append([phase, epoch, epoch_loss, epoch_acc.item()])
            experiment.log_metrics(
                {f"{phase}_loss": epoch_loss, f"{phase}_acc": epoch_acc.item()},
                epoch=epoch,
            )
            if epoch > 10:
                if phase == "val" and epoch_acc > best_acc:
                    print(f"New leading accuracy: {epoch_acc:.4f}")
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
            elif phase == "val":
                best_acc = epoch_acc
            try:
                experiment.log_metrics(
                    {
                        "best_val_loss": best_loss,
                        "best_val_acc": best_acc.item(),
                    },
                    epoch=epoch,
                )
            except AttributeError:
                experiment.log_metrics(
                    {
                        "best_val_loss": best_loss,
                        "best_val_acc": best_acc,
                    },
                    epoch=epoch,
                )
    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))
    model.load_state_dict(best_model_wts)
    training_results = pd.DataFrame(training_results)
    training_results.columns = ["phase", "epoch", "loss", "accuracy"]
    return model, training_results


def custom_load(batch_size, num_workers, train_df, val_df, test_df, image_dir):
    dataset_sizes = {"train": len(train_df), "val": len(val_df), "test": len(test_df)}

    transformed_train = SkinDataset(
        df=train_df,
        root_dir=image_dir,
        transform=transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=224),  # ImageNet statistics
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    )
    transformed_val = SkinDataset(
        df=val_df,
        root_dir=image_dir,
        transform=transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    )
    transformed_test = SkinDataset(
        df=test_df,
        root_dir=image_dir,
        transform=transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    )
    dataloaders = {
        "train": torch.utils.data.DataLoader(
            transformed_train,
            batch_size=32,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
        ),
        "val": torch.utils.data.DataLoader(
            transformed_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test": torch.utils.data.DataLoader(
            transformed_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }

    return dataloaders, dataset_sizes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument(
        "--optimizer", type=str, choices=["SGD", "Adam"], help="Optimizer"
    )
    parser.add_argument("--base_lr", type=float, help="Base learning rate")
    parser.add_argument(
        "--dev_mode",
        type=str,
        choices=["dev", "full"],
        help="dev mode (1000 images) or full mode (all images)",
    )
    parser.add_argument(
        "--data_list_file", type=str, help="Path to CSV file containing list of images"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        help="Path to directory containing all the images. Must be a flat directory with no subdirectories, containing the original file names.",
    )
    parser.add_argument(
        "--holdout_set",
        type=str,
        choices=[
            "expert_select",
            "random_holdout",
            "a12",
            "a34",
            "a56",
            "dermaamin",
            "br",
        ],
        help="Holdout set",
    )
    parser.add_argument(
        "--output_dir",
        default="./artefacts",
        type=str,
        help="Path to directory to save model and training results",
    )
    parser.add_argument("--seed", type=int, help="Random seed")
    flags = parser.parse_args()
    N_EPOCHS = flags.n_epochs
    OPTIMIZER = flags.optimizer
    BASE_LR = flags.base_lr
    DEV_MODE = flags.dev_mode
    DATA_LIST_FILE = flags.data_list_file
    IMAGES_DIR = flags.images_dir
    SEED = flags.seed
    HOLDOUT_SET = flags.holdout_set
    OUTPUT_DIR = flags.output_dir

    # Set the experiment name for Comet.ml logging
    experiment.set_name(f"{N_EPOCHS}_{OPTIMIZER}_{BASE_LR}_{HOLDOUT_SET}_seed_{SEED}")
    experiment.log_parameters(flags)

    # Creating the output directory and the experiment directory
    EXPERIMENT_DIR = os.path.join(
        OUTPUT_DIR, f"{N_EPOCHS}_{OPTIMIZER}_{BASE_LR}/{HOLDOUT_SET}_seed_{SEED}"
    )
    if not os.path.exists(EXPERIMENT_DIR):
        os.makedirs(EXPERIMENT_DIR)

    # Setting up the accelerator
    device = accelerator.device

    # set seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # https://twitter.com/subramen/status/1501231364879699969
    torch.set_flush_denormal(True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if DEV_MODE == "dev":
        df = pd.read_csv(DATA_LIST_FILE, header="infer").sample(1000)
    else:
        df = pd.read_csv(DATA_LIST_FILE, header="infer")

    train_df, val_df, test_df = get_partitions(DATA_LIST_FILE, HOLDOUT_SET)

    for indexer, label in enumerate(["low"]):
        weights = np.array(
            max(train_df[label].value_counts())
            / train_df[label].value_counts().sort_index()
        )
        label_codes = sorted(list(train_df[label].unique()))
        dataloaders, dataset_sizes = custom_load(
            batch_size=32,
            num_workers=20,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            image_dir=IMAGES_DIR,
        )

        model_ft = models.vgg16(pretrained=True)
        for param in model_ft.parameters():
            param.requires_grad = False

        model_ft.classifier[6] = nn.Sequential(
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Linear(256, len(label_codes)),
            nn.LogSoftmax(dim=1),
        )

        class_weights = torch.FloatTensor(weights).cuda()
        criterion = nn.NLLLoss()

        if OPTIMIZER == "SGD":
            optimizer_ft = optim.SGD(
                model_ft.parameters(), lr=BASE_LR, weight_decay=0.001, momentum=0.9
            )
        elif OPTIMIZER == "Adam":
            optimizer_ft = optim.Adam(
                model_ft.parameters(), lr=BASE_LR, weight_decay=0.001
            )
        else:
            raise ValueError("Invalid optimizer")

        # Prepare the model, the optimizer, and the dataloaders for HF Accelerate.
        (
            model_ft,
            optimizer_ft,
            dataloaders["train"],
            dataloaders["val"],
            dataloaders["test"],
        ) = accelerator.prepare(
            model_ft,
            optimizer_ft,
            dataloaders["train"],
            dataloaders["val"],
            dataloaders["test"],
        )

        print("\n........ Training ........ \n")
        model_ft, training_results = train_model(
            label,
            dataloaders,
            device,
            dataset_sizes,
            model_ft,
            criterion,
            optimizer_ft,
            N_EPOCHS,
        )
        print("\nTraining Complete")

        # https://huggingface.co/docs/accelerate/v0.25.0/en/package_reference/accelerator#saving-and-loading
        # torch.save(model_ft.state_dict(), "model_path_{}_{}_{}.pth".format(N_EPOCHS, label, HOLDOUT_SET))
        accelerator.wait_for_everyone()
        accelerator.save(
            model_ft.state_dict(),
            f"{EXPERIMENT_DIR}/model_path_{N_EPOCHS}_{label}_{HOLDOUT_SET}.pth",
        )

        print("........ Saving ........ \n")
        training_results.to_csv(
            f"{EXPERIMENT_DIR}/training_{N_EPOCHS}_{label}_{HOLDOUT_SET}.csv"
        )
        model = model_ft.eval()
        loader = dataloaders["val"]
        prediction_list = []
        diagcode_list = []
        fitzpatrick_list = []
        hasher_list = []
        labels_list = []
        p_list = []
        topk_p = []
        topk_n = []
        d1 = []
        d2 = []
        d3 = []
        p1 = []
        p2 = []
        p3 = []

        # Test the model
        print("........ Testing ........ \n")
        with torch.no_grad():
            running_corrects = 0
            for _, batch in enumerate(dataloaders["test"]):
                inputs = batch["image"].to(device)
                classes = batch[label].to(device)
                diagcode = batch["diagcode"]
                fitzpatrick = batch["fitzpatrick"]
                hasher = batch["hasher"]
                outputs = model(inputs.float())
                probability = outputs
                ppp, preds = torch.topk(probability, 1)
                if label == "low":
                    _, preds5 = torch.topk(probability, 3)
                    topk_p.append(np.exp(_.cpu()).tolist())
                    topk_n.append(preds5.cpu().tolist())

                p_list.append(ppp.cpu().tolist())
                prediction_list.append(preds.cpu().tolist())
                labels_list.append(classes.tolist())
                diagcode_list.append(diagcode)
                fitzpatrick_list.append(fitzpatrick.tolist())
                hasher_list.append(hasher)

        if label == "low":
            for j in topk_n:
                for i in j:
                    d1.append(i[0])
                    d2.append(i[1])
                    d3.append(i[2])
            for j in topk_p:
                for i in j:
                    p1.append(i[0])
                    p2.append(i[1])
                    p3.append(i[2])
            df_x = pd.DataFrame(
                {
                    "hasher": flatten(hasher_list),
                    "label": flatten(labels_list),
                    "diagcode": flatten(diagcode_list),
                    "fitzpatrick": flatten(fitzpatrick_list),
                    "prediction_probability": flatten(p_list),
                    "prediction": flatten(prediction_list),
                    "d1": d1,
                    "d2": d2,
                    "d3": d3,
                    "p1": p1,
                    "p2": p2,
                    "p3": p3,
                }
            )
        else:
            df_x = pd.DataFrame(
                {
                    "hasher": flatten(hasher_list),
                    "label": flatten(labels_list),
                    "diagcode": flatten(diagcode_list),
                    "fitzpatrick": flatten(fitzpatrick_list),
                    "prediction_probability": flatten(p_list),
                    "prediction": flatten(prediction_list),
                }
            )
        df_x.to_csv(
            f"{EXPERIMENT_DIR}/results_{N_EPOCHS}_{label}_{HOLDOUT_SET}.csv",
            index=False,
        )
        test_acc = (df_x.label == df_x.prediction).sum() / len(df_x)
        print(f"Overall test accuracy: {test_acc:.4f}")
        experiment.log_metric("overall_test_acc", test_acc)
        for fst in range(1, 7):
            if fst in df_x.fitzpatrick.unique():
                filtered_df_x = df_x[df_x.fitzpatrick == fst]
                filtered_df_x_acc = (
                    filtered_df_x.label == filtered_df_x.prediction
                ).sum() / len(filtered_df_x)
                print(f"Test accuracy for {fst}: {filtered_df_x_acc:.4f}")
                experiment.log_metric(f"FST_{fst}_test_acc", filtered_df_x_acc)
            else:
                print(f"Test accuracy for {fst}: N/A")
    print("=" * 30)
