import argparse
import os

import torch
import yaml
from ignite.contrib import metrics

import constants as const
import dataset
import fastflow
import utils
import sd

def build_train_data_loader(args, config):
    train_dataset = dataset.MVTecDataset(
        root=args.data,
        category=args.category,
        input_size=config["input_size"],
        is_train=True,
    )
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=const.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        drop_last=True,
    )

def build_train_ng_data_loader(args, config):
    train_dataset = dataset.MVTecDataset(
        root=args.data,
        category=args.category,
        input_size=config["input_size"],
        is_train=False,
        is_train_ng=True,
    )
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=const.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        drop_last=True,
    )    


def build_test_data_loader(args, config):
    test_dataset = dataset.MVTecDataset(
        root=args.data,
        category=args.category,
        input_size=config["input_size"],
        is_train=False,
    )
    return torch.utils.data.DataLoader(
        test_dataset,
        batch_size=const.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )


def build_model(config,sd_dim):
    model = fastflow.FastFlow(
        backbone_name=config["backbone_name"],
        flow_steps=config["flow_step"],
        input_size=config["input_size"],
        sd_dim=sd_dim,
        conv3x3_only=config["conv3x3_only"],
        hidden_ratio=config["hidden_ratio"],        
    )
    print(
        "Model A.D. Param#: {}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )
    return model


def build_optimizer(model):
    return torch.optim.Adam(
        model.parameters(), lr=const.LR, weight_decay=const.WEIGHT_DECAY
    )


def train_one_epoch(dataloader, model, optimizer, epoch, sd_dim):
    model.train()
    loss_meter = utils.AverageMeter()
    for step, data in enumerate(dataloader):
        # forward
        data = data.to("mps") #data = data.cuda()
        ret = model(data, sd_dim)
        loss = ret["loss"]
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # log
        loss_meter.update(loss.item())
        if (step + 1) % const.LOG_INTERVAL == 0 or (step + 1) == len(dataloader):
            print(
                "Epoch {} - Step {}: loss = {:.3f}({:.3f})".format(
                    epoch + 1, step + 1, loss_meter.val, loss_meter.avg
                )
            )


def eval_once(dataloader, model, sd_dim):
    model.eval()
    auroc_metric = metrics.ROC_AUC()
    for data, targets in dataloader:
        data, targets = data.to("mps"), targets.to("mps") #data, targets = data.cuda(), targets.cuda()
        with torch.no_grad():
            ret = model(data, sd_dim)
        outputs = ret["anomaly_map"].cpu().detach()
        outputs = outputs.flatten()
        targets = targets.flatten()
        auroc_metric.update((outputs, targets))
    auroc = auroc_metric.compute()
    print("AUROC: {}".format(auroc))

def train(args):
    os.makedirs(const.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_dir = os.path.join(
        const.CHECKPOINT_DIR, "exp%d" % len(os.listdir(const.CHECKPOINT_DIR))
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    config = yaml.safe_load(open(args.config, "r"))
    
    train_dataloader = build_train_data_loader(args, config)
    test_dataloader = build_test_data_loader(args, config)

    #Statistical Dimensionality Reduction
    #sd.defetcGen(args.data, args.category)
    #train_ng_dataloader = build_train_ng_data_loader(args,config)
    #sd_dim = sd.get_sd(config["backbone_name"], 100, train_dataloader, train_ng_dataloader)
    sd_dim = ['0-0', '0-10', '0-13', '0-14', '0-15', '0-16', '0-17', '0-18', '0-19', '0-2', '0-20', '0-21', '0-23', '0-24', '0-25', '0-26', '0-27', '0-29', '0-31', '0-33', '0-34', '0-35', '0-37', '0-38', '0-4', '0-41', '0-42', '0-44', '0-45', '0-46', '0-47', '0-48', '0-5', '0-51', '0-52', '0-53', '0-54', '0-55', '0-56', '0-57', '0-58', '0-59', '0-6', '0-60', '0-61', '0-62', '0-7', '0-8', '0-9', '1-102', '1-105', '1-107', '1-108', '1-109', '1-110', '1-113', '1-116', '1-120', '1-121', '1-18', '1-20', '1-21', '1-24', '1-26', '1-29', '1-32', '1-39', '1-41', '1-48', '1-49', '1-5', '1-54', '1-7', '1-76', '1-85', '1-88', '1-9', '1-98', '1-99', '2-111', '2-123', '2-130', '2-132', '2-140', '2-163', '2-180', '2-190', '2-2', '2-209', '2-210', '2-214', '2-217', '2-226', '2-238', '2-252', '2-27', '2-43', '2-52', '2-53', '2-6']
    
    model = build_model(config, sd_dim)
    optimizer = build_optimizer(model)

    model.to("mps") #model.cuda()

    for epoch in range(const.NUM_EPOCHS):
        train_one_epoch(train_dataloader, model, optimizer, epoch, sd_dim)
        if (epoch + 1) % const.EVAL_INTERVAL == 0:
            eval_once(test_dataloader, model, sd_dim)
        if (epoch + 1) % const.CHECKPOINT_INTERVAL == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(checkpoint_dir, "%d.pt" % epoch),
            )


def evaluate(args):
    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_dataloader = build_test_data_loader(args, config)
    model.to("mps") #model.cuda()
    eval_once(test_dataloader, model)


def parse_args():
    parser = argparse.ArgumentParser(description="Train FastFlow on MVTec-AD dataset")
    parser.add_argument(
        "-cfg", "--config", type=str, required=True, help="path to config file"
    )
    parser.add_argument("--data", type=str, required=True, help="path to mvtec folder")
    parser.add_argument(
        "-cat",
        "--category",
        type=str,
        choices=const.MVTEC_CATEGORIES,
        required=True,
        help="category name in mvtec",
    )
    parser.add_argument("--eval", action="store_true", help="run eval only")
    parser.add_argument(
        "-ckpt", "--checkpoint", type=str, help="path to load checkpoint"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.eval:
        evaluate(args)
    else:
        train(args)
