import torch
import argparse 
from collections import OrderedDict
from torch.nn import DataParallel
import os
from modeling.deeplab import *
def generate_onnx(model_path, onnx_path):
        checkpoint = torch.load(model_path)
        model = DeepLab(num_classes=2)
        model = model.cuda().eval()
        model.load_state_dict(checkpoint['state_dict'])

        dummy_input = torch.randn(1, 3, 512, 512, device='cuda')
        print('start exporting onnx model')
        torch.onnx.export(model, dummy_input, onnx_path, verbose=True)
        print('finish exporting')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DeeplabV3Plus")
    parser.add_argument('--model', type=str, default=None, help='trained model path')
    parser.add_argument('--onnx', type=str, default="./deeplab.onnx", help='onnx store path')
    args = parser.parse_args() 
    generate_onnx(args.model, args.onnx)
