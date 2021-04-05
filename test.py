from __future__ import division

# Import models and utils
from models import *
from utils.utils import *
from utils.datasets import *
from utils.augmentations import *
from utils.transforms import *
from utils.parse_config import *

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

# Function to calculate precision, recall, AP, f1 ...
def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    # Change to eval mode
    model.eval()

    # Get dataloader
    # Initialize dataset
    dataset = ListDataset(path, img_size=img_size, multiscale=False, transform=DEFAULT_TRANSFORMS)
    # Initialize data loader
    dataloader = torch.utils.data.DataLoader(
        dataset,                            # Set dataset
        batch_size=batch_size,              # Set batch size 
        shuffle=False,                      # Use shuffle (the data reshuffled at every epoch)
        num_workers=1,                      # Set how many subprocesses to use for data loading
        collate_fn=dataset.collate_fn       # Set collate_fn function (merging a list of samples)
    )

    # Initialize to available tensor type
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # Initialize label list and metric list
    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    # Iterate mini batch
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        
        if targets is None:
            continue
            
        # Extract labels to list
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        # Initialize image
        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        # Set no gradient (not tracking history)
        with torch.no_grad():
            # Inference model
            outputs = to_cpu(model(imgs))
            # execute non-max suppression
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, iou_thres=nms_thres)

        # Calculate sample metrics
        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)
    
    if len(sample_metrics) == 0:  # no detections over whole validation set.
        return None
    
    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    # Make parser instance and Read arguments 
    # Initialize argument parser
    parser = argparse.ArgumentParser()
    # Add arguments
    # mini-batch size
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")   
    # path to model definition
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file") 
    # path to config for data
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")     
    # path to model weight file
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")  
    # path to output label file
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")       
    # iou threshold for detection
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")   
    # confidence threshold for nms
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")                    
    # iou threshold for nms
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")      
    # number of cpu we use
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")    
    # image size
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")   
    # Read arguments
    opt = parser.parse_args()
    print(opt)

    # Check cuda availability and read device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    # Read path of validation data
    valid_path = data_config["valid"]
    # Load name of output classes
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    # If weight file extension is .weights
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    print("Compute mAP...")

    # Calculate precision, recall, AP ...
    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=opt.batch_size,
    )

    print("Average Precisions:")
    # Calculate Average Precisions
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")
