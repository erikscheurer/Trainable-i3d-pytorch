import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from src.i3dpt import Unit3Dpy
from utils.temporal_transforms import TemporalRandomCrop
from utils.utils import *
from src.i3dpt import I3D
from DataLoader import RGBFlowDataset
from opts import parser
import sys
sys.path.append('FlowUnderAttack/flow_library')
sys.path.append('FlowUnderAttack')
sys.path.append('FlowUnderAttack/helper_functions')
from FlowUnderAttack.helper_functions.ownutilities import show_images
import time
import os
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
def train_model(models, criterion, optimizers, schedulers, num_epochs=40, save_folder="", phases=["train", "val"]):
    since = time.time()
    # unique name for each run (using timestamp and learning rate)
    t = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    t += f"_lr={optimizers['flow'].param_groups[0]['lr']}"
    print(f"Saving to {os.path.join('runs', t)}", "batch_size=", data_loaders["train"].batch_size)
    writer = SummaryWriter(os.path.join("runs", t))
    
    best_model_wts = {}
    for stream in streams:
        best_model_wts[stream] = copy.deepcopy(models[stream].state_dict())

    best_accs = {stream: 0.0 for stream in streams}
    global_step = 0
    for epoch in range(num_epochs):
        log('Epoch {}/{}'.format(epoch, num_epochs - 1))
        log('-' * 10)

        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == 'train':
                [i.train() for i in models.values()]  # Set model to training mode
                running_losses = {"rgb": 0.0, "flow": 0.0}
                running_corrects = {"rgb": 0, "flow": 0}
            else:
                [i.eval() for i in models.values()]  # Set model to evaluate mode
                running_losses = {"rgb": 0.0, "flow": 0.0, "composed": 0.0}
                running_corrects = {"rgb": 0, "flow": 0, "composed": 0}

            # Iterate over data.
            data = {}
            for data["rgb"], data["flow"], labels in tqdm(data_loaders[phase],total=len(data_loaders[phase])):
                for stream in streams:
                    data[stream] = data[stream].to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                for optimizer in optimizers.values():
                    optimizer.zero_grad()

                # forward
                # track history if only in train
                out_logits = {}
                losses = {}
                with torch.set_grad_enabled(phase == 'train'):
                    # Calculate the joint output of two model
                    for stream in streams: # rgb and flow
                        if stream == "flow": # normalize to -1~1
                            data[stream] = data[stream] / data[stream].abs().max()

                        _, out_logits[stream] = models[stream](data[stream])
                        out_softmax = torch.nn.functional.softmax(out_logits[stream], 1)
                        _, preds = torch.max(out_softmax.data.cpu(), 1)
                        losses[stream] = criterion(out_softmax.cpu(), labels.cpu())

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            writer.add_scalar(f"Training loss: {stream}", losses[stream]/data[stream].shape[0], global_step) # normalize loss by batch size
                            losses[stream].backward()
                            optimizers[stream].step()
                            writer.add_scalar(f"Learning rate: {stream}", optimizers[stream].param_groups[0]['lr'], global_step)
                        if phase == "val":
                            writer.add_scalar(f"Validation loss: {stream}", losses[stream]/data[stream].shape[0], global_step)
                        if phase =='test':
                            writer.add_scalar(f"Test loss: {stream}", losses[stream]/data[stream].shape[0], global_step)

                        # statistics
                        running_losses[stream] += losses[stream].item() * data[stream].shape[0]
                        running_corrects[stream] += torch.sum(preds == labels.data.cpu())
            
                    if phase == "val":
                        out_logits["composed"] = sum([out_logits[stream] for stream in streams])
                        out_softmax = torch.nn.functional.softmax(out_logits["composed"], 1)
                        _, preds = torch.max(out_softmax.data.cpu(), 1)
                        losses["composed"] = criterion(out_softmax.cpu(), labels.cpu())
                        running_losses["composed"] += losses["composed"].item() * data[stream].shape[0]
                        running_corrects["composed"] += torch.sum(preds == labels.data.cpu())
                    
                    if phase=='train':
                        for scheduler in schedulers.values():
                            if isinstance(scheduler, lr_scheduler.OneCycleLR):
                                scheduler.step()
                        global_step+=1
            
            if phase=='val':
                for scheduler in schedulers.values():
                    if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(running_losses["composed"])
            if phase == 'train':
                for scheduler in schedulers.values():
                    if isinstance(scheduler, lr_scheduler.StepLR):
                        scheduler.step()

            epoch_losses = {}
            epoch_accs = {}
            for stream in losses.keys():
                epoch_losses[stream] = running_losses[stream] / dataset_sizes[phase]
                epoch_accs[stream] = running_corrects[stream].double() / dataset_sizes[phase]
                log('{} Loss ({}): {:.4f} Acc: {:.4f}'.format(
                    phase, stream, epoch_losses[stream], epoch_accs[stream]))

            # deep copy the model
            for stream in streams:
                if epoch_accs[stream] > best_accs[stream]: # phase == 'val' and 
                    best_accs[stream] = epoch_accs[stream]
                    best_model_wts[stream] = copy.deepcopy(models[stream].state_dict())
                    # save model
                    t = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
                    os.makedirs(os.path.join("model", save_folder), exist_ok=True)
                    torch.save(models[stream].state_dict(), os.path.join("model", save_folder, f"{stream}_best_model_{t}.pth"))

    time_elapsed = time.time() - since
    log('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    for stream in streams:
        log('Best val Acc({}): {:4f}'.format(stream, best_accs[stream]))

    # load best model weights
    for stream in streams:
        models[stream].load_state_dict(best_model_wts[stream])
    return models


def load_and_freeze_model(model, weight_path, num_classes, num_freeze=15):
    model.load_state_dict(torch.load(weight_path))
    log("Pre-trained model {} loaded successfully!".format(weight_path))
    counter = 0
    for child in model.children():
        counter += 1
        if counter < num_freeze:
            log("Layer {} frozen!".format(child._get_name()))
            for param in child.parameters():
                param.requires_grad = False
    model.num_classes = num_classes
    if num_classes == 400:
        log("Changing the output layer for fine-tuning...")
        model.conv3d_0c_1x1 = Unit3Dpy(in_channels=1024,
                                    out_channels=num_classes,
                                    kernel_size=(1, 1, 1),
                                    activation=None,
                                    use_bias=True,
                                    use_bn=False)
    model.to(device)


def main():
    weight_paths = {"rgb": args.rgb_weights_path, "flow": args.flow_weights_path}
    optimizers = {}
    schedulers = {}

    if weight_paths["flow"] == 'model/model_flow.pth':
        # Here we must use 400 num_class because we have to load the weight from original file. We change it later.
        num_classes = 400
    else:
        num_classes = len(class_names)
    models = {x: I3D(num_classes=num_classes, modality=x) for x in streams}

    # freeze the first 15 layers
    for stream in streams:
        load_and_freeze_model(model=models[stream], num_classes=len(class_names), weight_path=weight_paths[stream], num_freeze=args.num_freeze)
        optimizers[stream] = optim.SGD(filter(lambda p: p.requires_grad, models[stream].parameters()), lr=.1,
                                       momentum=0.9)
        # schedulers[stream] = lr_scheduler.StepLR(optimizers[stream], step_size=10, gamma=0.5)
        # schedulers[stream] = lr_scheduler.OneCycleLR(optimizers[stream], max_lr=0.01, steps_per_epoch=len(data_loaders["train"]), epochs=50)
        schedulers[stream] = lr_scheduler.ReduceLROnPlateau(optimizers[stream], mode='min', factor=0.1, patience=15, verbose=True)

    criterion = nn.CrossEntropyLoss()
    train_model(models, criterion, optimizers, schedulers, num_epochs=500, save_folder=args.save_folder, phases=['train', 'val'])
    # test_model(models, criterion, phases=['test'])
    train_model(models, criterion, optimizers, schedulers, num_epochs=1, save_folder=args.save_folder, phases=['test'])
    for stream in streams:
        temp_model_path = 'model/{}_model_{}.pth'.format(args.session_id, stream)
        torch.save(models[stream].state_dict(), temp_model_path)


if __name__ == "__main__":

    args = parser.parse_args()
    split_dir = "data/ucf101_splits"
    class_names = [i.strip() for i in open(args.classes_path)]
    class_dicts = {k: v for v, k in enumerate(class_names)}
    data_dir = Path(args.data_path)
    streams = []
    if args.rgb:
        streams.append('rgb')
    if args.flow:
        streams.append('flow')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    rgb_flow_datasets = {x: RGBFlowDataset(data_dir, 
                                           split_dir,
                                           split=x,
                                           class_dict = class_dicts,
                                           sample_rate=args.sample_num,
                                           sample_type=args.sample_type,
                                           fps=args.out_fps,
                                           out_frame_num=args.out_frame_num,
                                           augment=(x == "train"))
                         for x in ['train', 'val', 'test']}
    data_loaders = {x: torch.utils.data.DataLoader(rgb_flow_datasets[x], batch_size=64,
                                                   shuffle=True, num_workers=0, 
                                                   pin_memory=True)
                    for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(rgb_flow_datasets[x]) for x in ['train', 'val', 'test']}
    main()
