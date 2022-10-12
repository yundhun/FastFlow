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
import random
import time
import wandb

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
        start_time = time.time()
        # forward
        data = data.to(const.TORCH_DEVICE) #data = data.cuda()
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
            wandb.log({'elapsed_time per epoch':time.time()-start_time})
            start_time = time.time()


def eval_once(dataloader, model, sd_dim):
    model.eval()
    auroc_metric = metrics.ROC_AUC()
    for data, targets in dataloader:
        data, targets = data.to(const.TORCH_DEVICE), targets.to(const.TORCH_DEVICE) #data, targets = data.cuda(), targets.cuda()
        with torch.no_grad():
            ret = model(data, sd_dim)
        outputs = ret["anomaly_map"].cpu().detach()
        outputs = outputs.flatten()
        targets = targets.flatten()
        auroc_metric.update((outputs, targets))
    auroc = auroc_metric.compute()
    print("AUROC: {}".format(auroc))
    wandb.log({'auroc':auroc})

def train(args, sd_dim_size, dr_type):
    os.makedirs(const.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_dir = os.path.join(
        const.CHECKPOINT_DIR, "exp%d" % len(os.listdir(const.CHECKPOINT_DIR))
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    config = yaml.safe_load(open(args.config, "r"))
    
    train_dataloader = build_train_data_loader(args, config)
    test_dataloader = build_test_data_loader(args, config)

    #[64, 128, 256]
    sd_dim_full = []
    for ii in range(64):
        sd_dim_full = sd_dim_full + ['0-'+str(ii)]
    for ii in range(128):
        sd_dim_full = sd_dim_full + ['1-'+str(ii)]
    for ii in range(256):
        sd_dim_full = sd_dim_full + ['2-'+str(ii)]        
    
    #Statistical Dimensionality Reduction
    if dr_type == 'SD' :
        if sd_dim_size == 100:
            sd.defetcGen(args.data, args.category)
        if sd_dim_size < 256:
            train_ng_dataloader = build_train_ng_data_loader(args,config)
            sd_dim = sd.get_sd(config["backbone_name"], sd_dim_size, train_dataloader, train_ng_dataloader)
        if sd_dim_size == 256:
            sd_dim = sd_dim_full
    elif dr_type == 'RD' :
        sd_dim = random.sample(sd_dim_full,sd_dim_size)
          
    
    model = build_model(config, sd_dim)
    optimizer = build_optimizer(model)

    model.to(const.TORCH_DEVICE) #model.cuda()

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
    model.to(const.TORCH_DEVICE) #model.cuda()
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
    for sd_dim_size in [100,128,200,256]:
        wandb.init(project=args.category, name=args.category + str(sd_dim_size) + '(sd)')
        if args.eval:
            evaluate(args)
        else:
            train(args, sd_dim_size, const.DR_TYPE)
