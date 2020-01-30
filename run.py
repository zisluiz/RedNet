import argparse
import torch
import imageio
import numpy as np
import scipy.misc as misc
import skimage.transform
import PIL.Image
import torchvision

import torch.optim
import RedNet_model
from utils import utils
from utils.utils import load_ckpt
import os
import os.path as osp
from pathlib import Path
import psutil
from datetime import datetime
import time
import nvidia_smi
import sys
import glob

parser = argparse.ArgumentParser(description='RedNet Indoor Sementic Segmentation')
parser.add_argument('-r', '--rgb', default=None, metavar='DIR',
                    help='path to image')
parser.add_argument('-d', '--depth', default=None, metavar='DIR',
                    help='path to depth')
parser.add_argument('-o', '--output', default=None, metavar='DIR',
                    help='path to output')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--last-ckpt', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

sys.argv.extend(['--cuda', '--last-ckpt', 'checkpoint/rednet_ckpt_ori.pth'])

args = parser.parse_args()
device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
image_w = 640
image_h = 480
def inference():
    os.makedirs('results', exist_ok=True)
    f = open("results/run_"+str(int(round(time.time() * 1000)))+".txt", "w+")
    f.write('=== Start time: '+str(datetime.now())+'\n')

    p = psutil.Process(os.getpid())
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

    model = RedNet_model.RedNet(pretrained=False)
    load_ckpt(model, None, args.last_ckpt, device)
    model.eval()
    model.to(device)

    print('Starting list image files')
    filesCount = 0

    files = glob.glob("datasets/mestrado/**/rgb/*.png", recursive=True)
    files.extend(glob.glob("datasets/mestrado/**/rgb/*.jpg", recursive=True))
    cpuTimes = [0.0, 0.0, 0.0, 0.0]

    gpuTimes = 0.0
    gpuMemTimes = 0.0
    maxNumThreads = 0
    memUsageTimes = 0

    for imagePath in files:
        print('imagePath: ' + imagePath)
        pathRgb = Path(imagePath)
        datasetName = osp.basename(str(pathRgb.parent.parent))
        # print('datasetName: ' + datasetName)
        parentDatasetDir = str(pathRgb.parent.parent)
        # print('parentDatasetDir: ' + parentDatasetDir)
        depthImageName = os.path.basename(imagePath).replace('jpg', 'png')

        image = imageio.imread(imagePath)
        depth = imageio.imread(parentDatasetDir + '/depth/' + depthImageName)

        if datasetName == "active_vision" or datasetName == "putkk":
            image = image[0:1080, 419:1499]
            depth = depth[0:1080, 419:1499]

        # Bi-linear
        image = skimage.transform.resize(image, (image_h, image_w), order=1,
                                         mode='reflect', preserve_range=True)
        # Nearest-neighbor
        depth = skimage.transform.resize(depth, (image_h, image_w), order=0,
                                         mode='reflect', preserve_range=True)

        image = image / 255
        image = torch.from_numpy(image).float()
        depth = torch.from_numpy(depth).float()
        image = image.permute(2, 0, 1)
        depth.unsqueeze_(0)

        image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])(image)
        depth = torchvision.transforms.Normalize(mean=[19050],
                                                 std=[9650])(depth)

        image = image.to(device).unsqueeze_(0)
        depth = depth.to(device).unsqueeze_(0)

        pred = model(image, depth)

        res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        curGpuTime = res.gpu
        #curGpuMemTime = res.memory #(in percent)
        curGpuMemTime = mem_res.used / 1e+6
        gpuTimes += curGpuTime
        gpuMemTimes += curGpuMemTime
        f.write('GPU Usage Percent: ' + str(curGpuTime) + '\n')
        f.write('GPU Mem Usage (MB)): ' + str(curGpuMemTime) + '\n')

        curProcessCpuPerU = p.cpu_percent()
        curCpusPerU = psutil.cpu_percent(interval=None, percpu=True)

        # gives a single float value
        for i in range(len(cpuTimes)):
            curProcessCpu = curProcessCpuPerU
            curCpu = curCpusPerU[i]
            cpuTimes[i] += curCpu
            f.write('Process CPU Percent: ' + str(curProcessCpu) + ' --- CPU Percent: ' + str(curCpu) + '\n')

        # you can convert that object to a dictionary
        memInfo = dict(p.memory_full_info()._asdict())
        curMemUsage = memInfo['uss']
        memUsageTimes += curMemUsage

        f.write('Process memory usage: ' + str(curMemUsage / 1e+6) + '\n')
        f.write('Memory information: ' + str(memInfo) + '\n')

        if maxNumThreads < p.num_threads():
            maxNumThreads = p.num_threads()

        # print('############## Index: ')
        # print(index)
        os.makedirs('results/' + datasetName, exist_ok=True)

        output = utils.to_label(torch.max(pred, 1)[1] + 1)
        #output = utils.to_label(torch.max(pred, 1)[1] + 1)[0]
        #imageio.imsave('results/' + datasetName + '/' + depthImageName, output.cpu().numpy().transpose((1, 2, 0)))
        #imageio.imsave('results/' + datasetName + '/' + depthImageName, output)
        lbl_pil = PIL.Image.fromarray(output.astype(np.uint8), mode='P')
        lbl_pil.save('results/' + datasetName + '/' + depthImageName)
        filesCount = filesCount + 1

        del image, depth, pred, output

        torch.cuda.empty_cache()
    nvidia_smi.nvmlShutdown()

    start = time.time()
    for imagePath in files:
        pathRgb = Path(imagePath)
        datasetName = osp.basename(str(pathRgb.parent.parent))
        parentDatasetDir = str(pathRgb.parent.parent)
        depthImageName = os.path.basename(imagePath).replace('jpg', 'png')

        image = imageio.imread(imagePath)
        depth = imageio.imread(parentDatasetDir + '/depth/' + depthImageName)

        if datasetName == "active_vision" or datasetName == "putkk":
            image = image[0:1080, 419:1499]
            depth = depth[0:1080, 419:1499]

        # Bi-linear
        image = skimage.transform.resize(image, (image_h, image_w), order=1,
                                         mode='reflect', preserve_range=True)
        # Nearest-neighbor
        depth = skimage.transform.resize(depth, (image_h, image_w), order=0,
                                         mode='reflect', preserve_range=True)

        image = image / 255
        image = torch.from_numpy(image).float()
        depth = torch.from_numpy(depth).float()
        image = image.permute(2, 0, 1)
        depth.unsqueeze_(0)

        image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])(image)
        depth = torchvision.transforms.Normalize(mean=[19050],
                                                 std=[9650])(depth)

        image = image.to(device).unsqueeze_(0)
        depth = depth.to(device).unsqueeze_(0)

        pred = model(image, depth)

        del image, depth, pred

        #torch.cuda.empty_cache()
    end = time.time()

    f.write('=== Mean GPU Usage Percent: ' + str(gpuTimes / filesCount) + '\n')
    f.write('=== Mean GPU Mem Usage (MB): ' + str(gpuMemTimes / filesCount) + '\n')
    for i in range(len(cpuTimes)):
        f.write("=== Mean cpu" + str(i) + " usage: " + str(cpuTimes[i] / filesCount) + '\n')
    f.write("=== Mean memory usage (MB): " + str((memUsageTimes / filesCount) / 1e+6) + '\n')

    f.write("=== Total image predicted: " + str(filesCount) + '\n')
    f.write("=== Seconds per image: " + str(((end - start) / filesCount)) + '\n')
    f.write("=== Max num threads: " + str(maxNumThreads) + '\n')

    f.write('=== End time: ' + str(datetime.now()) + '\n')
    f.close()

if __name__ == '__main__':
    inference()
