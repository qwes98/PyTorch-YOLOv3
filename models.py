from __future__ import division
from itertools import chain

# Import torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# Import utils
from utils.parse_config import *
from utils.utils import build_targets, to_cpu, non_max_suppression

# Import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Make layers from module definition on conf file
def create_modules(module_defs):
    # Get hyperparameters
    hyperparams = module_defs.pop(0)
    # Update hyperparameters to correct type
    hyperparams.update({
        'batch': int(hyperparams['batch']),                 # size of batch (mini-batch)
        'subdivisions': int(hyperparams['subdivisions']),   # size of subdivisions (split mini-batch)
        'width': int(hyperparams['width']),                 # input image width
        'height': int(hyperparams['height']),               # input image height
        'channels': int(hyperparams['channels']),           # size of image channel
        'optimizer': hyperparams.get('optimizer'),          # name of optimizer
        'momentum': float(hyperparams['momentum']),         # hyperparameters for momentum (this will be used in optimizataion)
        'decay': float(hyperparams['decay']),               # hyperparameters for weight-decay
        'learning_rate': float(hyperparams['learning_rate']),   # learning rate
        'burn_in': int(hyperparams['burn_in']),             # hyperparameters for burn in (decrease learning rate)
        'max_batches': int(hyperparams['max_batches']),     # max value of batches
        'policy': hyperparams['policy'],                    # decide policy (what will be used in training)
        'lr_steps': list(zip(map(int,   hyperparams["steps"].split(",")),   # learning steps threshold 
                             map(float, hyperparams["scales"].split(",")))) # scale value for decreasing learning rate
    })
    # Check image width and height are same
    assert hyperparams["height"] == hyperparams["width"], \
        "Height and width should be equal! Non square images are padded with zeros."
    # Set number of channels
    output_filters = [hyperparams["channels"]]
    # Initialize module list
    module_list = nn.ModuleList()
    # Iterate and define module_list
    for module_i, module_def in enumerate(module_defs):
        # Initialize layer module
        modules = nn.Sequential()

        # If module type is convolution
        if module_def["type"] == "convolutional":
            # Set whether adding batch normalization or not
            bn = int(module_def["batch_normalize"])
            # Set number of filters
            filters = int(module_def["filters"])
            # Set kernal size
            kernel_size = int(module_def["size"])
            # Set padding
            pad = (kernel_size - 1) // 2
            # Define module
            modules.add_module(
                f"conv_{module_i}",                     # Set layer name
                nn.Conv2d(                              
                    in_channels=output_filters[-1],     # Set layer input feature map channel
                    out_channels=filters,               # Set layer output feature map channel
                    kernel_size=kernel_size,            # Set kernel(filter) size
                    stride=int(module_def["stride"]),   # Set stride (how many we will move kernel)
                    padding=pad,                        # Set padding
                    bias=not bn,                        # Set bias
                ),
            )
            # If use batch normalization
            if bn:
                # Add batch norm layer
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            # If activation is set to leaky
            if module_def["activation"] == "leaky":
                # Add leaky for activation layer
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))

        # If module type is maxpool
        elif module_def["type"] == "maxpool":
            # Read kernel size
            kernel_size = int(module_def["size"])
            # Read stride
            stride = int(module_def["stride"])
            # If kernal size is 2 and stride is 1
            if kernel_size == 2 and stride == 1:
                # Add zero padding layer
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            # Define maxpool layer
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            # Add maxpool layer
            modules.add_module(f"maxpool_{module_i}", maxpool)

        # If module type is upsample
        elif module_def["type"] == "upsample":
            # Define upsample layer
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            # Add upsample layer
            modules.add_module(f"upsample_{module_i}", upsample)

        # If module type is route
        elif module_def["type"] == "route":
            # Get layers
            layers = [int(x) for x in module_def["layers"].split(",")]
            # Define filter
            filters = sum([output_filters[1:][i] for i in layers])
            # Add route layer
            modules.add_module(f"route_{module_i}", nn.Sequential())

        # If module type is shortcut
        elif module_def["type"] == "shortcut":
            # Define filter
            filters = output_filters[1:][int(module_def["from"])]
            # Add shoutcut layer
            modules.add_module(f"shortcut_{module_i}", nn.Sequential())

        # If module type is yolo
        elif module_def["type"] == "yolo":
            # Get anchor index
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            # Set number of classes
            num_classes = int(module_def["classes"])
            # Set image size
            img_size = int(hyperparams["height"])
            # Set ignore threshold
            ignore_thres = float(module_def["ignore_thresh"])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_size, ignore_thres)
            # Add yolo layer
            modules.add_module(f"yolo_{module_i}", yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


# Define Upsample Layer 
class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x

# Yolo Layer class
class YOLOLayer(nn.Module):

    def __init__(self, anchors, num_classes, img_size, ignore_thres):
        super(YOLOLayer, self).__init__()
        # Set number of anchors (denote how many object could be detected in one grid cell)
        self.num_anchors = len(anchors)
        # Set number of classes
        self.num_classes = num_classes
        # Set ignore threshold
        self.ignore_thres = 0.5
        # Set MSE(Mean Squared Error) Loss
        self.mse_loss = nn.MSELoss()
        # Set BCE(Binary Cross Entropy) Loss
        self.bce_loss = nn.BCELoss()
        # Set number of outputs per anchor
        self.no = num_classes + 5 
        # Set grid
        self.grid = torch.zeros(1) # TODO

        # Set anchor
        anchors = torch.tensor(list(chain(*anchors))).float().view(-1, 2)
        # Register buffer
        self.register_buffer('anchors', anchors)
        self.register_buffer('anchor_grid', anchors.clone().view(1, -1, 1, 1, 2))
        # Set image size
        self.img_size = img_size
        # Initialize stride
        self.stride = None

    # forward method
    def forward(self, x):
        # Set stride
        stride = self.img_size // x.size(2)
        # Apply stride value to instance var
        self.stride = stride
        # Get feature map shape
        bs, _, ny, nx = x.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        x = x.view(bs, self.num_anchors, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        # If inference time
        if not self.training:  # inference
            # If not same with grid shape and x shape
            if self.grid.shape[2:4] != x.shape[2:4]:
                # Make grid
                self.grid = self._make_grid(nx, ny).to(x.device)

            # Make sigmoid
            y = x.sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid.to(x.device)) * stride  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid  # wh
            y = y.view(bs, -1, self.no)

        return x if self.training else y

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


# YOLO object detection model
class Darknet(nn.Module):

    # Constructor
    def __init__(self, config_path, img_size=416):
        # Call Module class constructor
        super(Darknet, self).__init__()

        # Initialize instance variables
        # Get module definition from config file
        self.module_defs = parse_model_config(config_path)
        # Get hyperparameters and 
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        # Initialize yolo model layers
        self.yolo_layers = [layer[0] for layer in self.module_list if isinstance(layer[0], YOLOLayer)]
        # Initialize image size
        self.img_size = img_size
        # Initialize seen variable
        self.seen = 0
        # Initialize head info
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    # Forward function
    def forward(self, x):
        # Define layer output and yolo output
        layer_outputs, yolo_outputs = [], []
        # Iterate with module definition and module list
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            # If module type is convolutional or upsample or maxpool
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                # Pass x to module
                x = module(x)
            # If type is route
            elif module_def["type"] == "route":
                # Do route
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            # If type is shortcut
            elif module_def["type"] == "shortcut":
                # Get before layer index
                layer_i = int(module_def["from"])
                # Do shortcut
                x = layer_outputs[-1] + layer_outputs[layer_i]
            # If type is yolo
            elif module_def["type"] == "yolo":
                # Pass x to module
                x = module[0](x)
                yolo_outputs.append(x)
            # Add output
            layer_outputs.append(x)
        return yolo_outputs if self.training else torch.cat(yolo_outputs, 1)

    # Parse and load darknet weight from path
    def load_darknet_weights(self, weights_path):

        # Open the weights file
        with open(weights_path, "rb") as f:
            # Parse header 
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            # Store header to instance variable
            self.header_info = header  # Needed to write header when saving weights
            # Store seen value
            self.seen = header[3]  # number of images seen during training
            # Read weights
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75

        # Initialize ptr
        ptr = 0
        # Iterate module definition and module list
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            # If cutoff
            if i == cutoff:
                break
            # If type is convolutional
            if module_def["type"] == "convolutional":
                # Get cont layer info
                conv_layer = module[0]
                # If do batch normalization
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    # Save darknet weights
    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        # Open write file
        fp = open(path, "wb")
        # Get seen
        self.header_info[3] = self.seen
        # Write header info to file
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            # If type is convolutional
            if module_def["type"] == "convolutional":
                # Get conv layer info
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    # Write batch normalization info to file
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        # Close file
        fp.close()
