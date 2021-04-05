from __future__ import division

# Import model and utils
from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.augmentations import *
from utils.transforms import *
from utils.parse_config import *
from utils.loss import compute_loss
from test import evaluate

from terminaltables import AsciiTable

# Import python modules
import os
import sys
import time
import datetime
import argparse
import tqdm

# Import torch modules
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


if __name__ == "__main__":
    # Make parser instance and Read arguments for training, logging ...
    # Initialize argument parser
    parser = argparse.ArgumentParser()
    # Add arguments
    # number of epochs
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs") 
    # path to read model cfg file
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file") 
    # path to data cfg file (# of output classes, data path, ...)
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file") 
    # path to pretrained weights file
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model") 
    # number of cpu
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation") 
    # size of input image
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension") 
    # interval time to save checkpoint 
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights") 
    # interval time to evaluate
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set") 
    # whether to use multi scale (multi scale makes image to different size)
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training") 
    # whether to show logging
    parser.add_argument("--verbose", "-v", default=False, action='store_true', help="Makes the training more verbose") 
    # path to save log file
    parser.add_argument("--logdir", type=str, default="logs", help="Defines the directory where the training log files are stored") 
    # Read arguments
    opt = parser.parse_args()
    print(opt)

    # Make logger
    logger = Logger(opt.logdir) 

    # Check cuda availability and read device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Make directories to save output and checkpoints
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    # Read path of train and validation data
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    # Load name of output classes
    class_names = load_classes(data_config["names"])

    # Initiate model on device
    model = Darknet(opt.model_def).to(device)
    # Apply weight initialize function to model
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        # If weight file extension is .pth
        if opt.pretrained_weights.endswith(".pth"):
            # Load pretrained weights from .pth file
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            # Load pretrained darknet weights 
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    # Initialize dataset
    dataset = ListDataset(train_path, multiscale=opt.multiscale_training, img_size=opt.img_size, transform=AUGMENTATION_TRANSFORMS)
    # Initialize dataloadaer
    dataloader = torch.utils.data.DataLoader(
        dataset,                                                                     # Set dataset
        batch_size= model.hyperparams['batch'] // model.hyperparams['subdivisions'], # Calculate and set batch size
        shuffle=True,                                                                # Use shuffle (the data reshuffled at every epoch)
        num_workers=opt.n_cpu,                                                       # Set how many subprocesses to use for data loading
        pin_memory=True,                                                             # Use pin memory (copying Tensors into CUDA pinned memory)
        collate_fn=dataset.collate_fn,                                               # Set collate_fn function (merging a list of samples)                                                   
    )

    # Choose optimizer 
    # If set optimizer to adam or not set (default)
    if (model.hyperparams['optimizer'] in [None, "adam"]):
        # Use Adam opimizer
        optimizer = torch.optim.Adam(
            model.parameters(),                         # Set parameters
            lr=model.hyperparams['learning_rate'],      # Set learning rate (how many we will go when optimizing)
            weight_decay=model.hyperparams['decay'],    # Set weight decay
            )
    elif (model.hyperparams['optimizer'] == "sgd"):
        # Use SGD(Sochastic Gradient Descent) optimizer
        optimizer = torch.optim.SGD(
            model.parameters(),                         # Set parameters
            lr=model.hyperparams['learning_rate'],      # Set learning rate (how many we will go when optimizing)
            weight_decay=model.hyperparams['decay'],    # Set weight decay hyperparameter (regularization by adding a small penalty)
            momentum=model.hyperparams['momentum'])     # Set momentum hyperparameter (using past gradient velocity)
    else:
        print("Unknown optimizer. Please choose between (adam, sgd).")

    # Iterate train by using training dataset as many times as epoch value
    for epoch in range(opt.epochs):
        print("\n---- Training Model ----")
        # Start to train
        model.train()
        # Store start time to check training time taken
        start_time = time.time()
        # Iterate mini batch
        for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch}")):
            # Calculate mini batch number
            # This is some sort of global mini batch index
            batches_done = len(dataloader) * epoch + batch_i

            # Load image to device memory
            imgs = imgs.to(device, non_blocking=True)
            # Load target values to device memory
            targets = targets.to(device)

            # Inference image through model
            outputs = model(imgs)

            # Compute loss
            loss, loss_components = compute_loss(outputs, targets, model)

            # Give calculated gradients in each layer
            # Accumulate backward values during subdivisions
            loss.backward()

            # Run optimizer

            # If our predetermined mini batch is finished
            if batches_done % model.hyperparams['subdivisions'] == 0:
                # Adapt learning rate
                # Get learning rate defined in cfg
                lr = model.hyperparams['learning_rate']
                # If time to burn in
                if batches_done < model.hyperparams['burn_in']:
                    # Burn in
                    lr *= (batches_done / model.hyperparams['burn_in'])
                else:
                    # Set and parse the learning rate to the steps defined in the cfg
                    for threshold, value in model.hyperparams['lr_steps']:
                        if batches_done > threshold:
                            lr *= value     # update learning rate
                # Log the learning rate
                logger.scalar_summary("train/learning_rate", lr, batches_done)
                # Set learning rate
                for g in optimizer.param_groups:
                    g['lr'] = lr

                # Run optimizer
                optimizer.step()
                # Reset gradients
                optimizer.zero_grad()

            # Log progress
            # Initialize log string
            log_str = ""
            # Logging
            log_str += AsciiTable(
                [
                    ["Type", "Value"],
                    ["IoU loss", float(loss_components[0])],
                    ["Object loss", float(loss_components[1])], 
                    ["Class loss", float(loss_components[2])],
                    ["Loss", float(loss_components[3])],
                    ["Batch loss", to_cpu(loss).item()],
                ]).table

            # Print log
            if opt.verbose: print(log_str)

            # Tensorboard logging
            tensorboard_log = [
                    ("train/iou_loss", float(loss_components[0])),
                    ("train/obj_loss", float(loss_components[1])), 
                    ("train/class_loss", float(loss_components[2])),
                    ("train/loss", to_cpu(loss).item())]
            # Loggin summary
            logger.list_of_scalars_summary(tensorboard_log, batches_done)

            # Update model seen value
            model.seen += imgs.size(0)

        # If time to evaluate model
        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            metrics_output = evaluate(
                model,                          # Set model
                path=valid_path,                # Set valid path             
                iou_thres=0.5,                  # Set IOU(Intersection Over Union) threshold (to remove bounding box for same target)
                conf_thres=0.1,                 # Set confidence threshold (This will be used to remove low confidence bounding box)
                nms_thres=0.5,                  # Set NMS(Non-Maximun Suppression) threshold 
                img_size=opt.img_size,          # Set image size
                batch_size=model.hyperparams['batch'] // model.hyperparams['subdivisions'],     # Set mini batch size by subdivision 
            )
            
            if metrics_output is not None:
                precision, recall, AP, f1, ap_class = metrics_output
                evaluation_metrics = [
                ("validation/precision", precision.mean()),
                ("validation/recall", recall.mean()),
                ("validation/mAP", AP.mean()),
                ("validation/f1", f1.mean()),
                ]
                logger.list_of_scalars_summary(evaluation_metrics, epoch)

                # If set to use verbose
                if opt.verbose:
                    # Print class APs and mAP
                    ap_table = [["Index", "Class name", "AP"]]
                    for i, c in enumerate(ap_class):
                        ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
                    print(AsciiTable(ap_table).table)
                    print(f"---- mAP {AP.mean()}")                
            else:
                print( "---- mAP not measured (no detections found by model)")

        # If time to store checkpoint
        if epoch % opt.checkpoint_interval == 0:
            # Save checkpoint
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
