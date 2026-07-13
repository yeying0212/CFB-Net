import sys
sys.path.insert(0, '.')
from models.cfb_net import CFBNet
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.parallel import gather
import torch.optim.lr_scheduler

import dataset as myDataLoader
import Transforms as myTransforms
from metric_tool import CDMetricsMeter
from PIL import Image

import time
import numpy as np
from argparse import ArgumentParser

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def BCEDiceLoss(inputs, targets):
    bce = F.binary_cross_entropy(inputs, targets)
    inter = (inputs * targets).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    return bce + 1 - dice


def BCE(inputs, targets):
    bce = F.binary_cross_entropy(inputs, targets)
    return bce

@torch.no_grad()
def val(args, val_loader, model, epoch):
    model.eval()

    salEvalVal = CDMetricsMeter(tolerance=5)
    epoch_loss = []
    total_batches = len(val_loader)
    print(len(val_loader))

    img_names = []
    img_metrics = []
    fft_vis_dir = os.path.join(args.vis_dir, 'fft_vis')

    # FPS timing
    total_infer_time = 0.0
    total_images = 0
    warmed_up = False

    for iter, batched_inputs in enumerate(val_loader):
        img, target = batched_inputs

        img_path = val_loader.sampler.data_source.file_list[iter]
        img_name = os.path.basename(img_path)
        img_names.append(img_name)

        pre_img = img[:, 0:3]
        post_img = img[:, 3:6]

        if args.onGPU == True:
            pre_img = pre_img.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            post_img = post_img.cuda(non_blocking=True)

        pre_img_var = torch.autograd.Variable(pre_img).float()
        post_img_var = torch.autograd.Variable(post_img).float()
        target_var = torch.autograd.Variable(target).float()

        # =========================
        # measure forward time only
        if args.onGPU:
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        # forward
        output, output2, output3, output4 = model(pre_img_var, post_img_var)

        if args.onGPU:
            torch.cuda.synchronize()

        infer_time = time.perf_counter() - t0

        # skip first warmup iteration
        if not warmed_up:
            warmed_up = True
        else:
            bs = pre_img_var.shape[0]
            total_infer_time += infer_time
            total_images += bs

        # current batch FPS
        bs = pre_img_var.shape[0]
        cur_fps = bs / max(infer_time, 1e-12)

        # loss computation (excluded from FPS timing)
        loss = BCEDiceLoss(output, target_var) + BCEDiceLoss(output2, target_var) + \
               BCEDiceLoss(output3, target_var) + BCEDiceLoss(output4, target_var)

        pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)).long()
        epoch_loss.append(loss.data.item())

        if args.onGPU and torch.cuda.device_count() > 1:
            output = gather(pred, 0, dim=0)

        pr = pred[0, 0].cpu().numpy()
        gt = target_var[0, 0].cpu().numpy()
        index_tp = np.where(np.logical_and(pr == 1, gt == 1))
        index_fp = np.where(np.logical_and(pr == 1, gt == 0))
        index_tn = np.where(np.logical_and(pr == 0, gt == 0))
        index_fn = np.where(np.logical_and(pr == 0, gt == 1))

        map = np.zeros([gt.shape[0], gt.shape[1], 3])
        map[index_tp] = [255, 255, 255]
        map[index_fp] = [255, 0, 0]
        map[index_tn] = [0, 0, 0]
        map[index_fn] = [0, 255, 0]

        change_map = Image.fromarray(np.array(map, dtype=np.uint8))
        change_map.save(args.vis_dir + img_name)

        per_img = salEvalVal.update(pr, gt)
        img_metrics.append(per_img)

        # average FPS (forward only)
        avg_fps = (total_images / total_infer_time) if total_infer_time > 0 else 0.0

        if iter % 5 == 0:
            bf1 = per_img['boundary_f1']
            print(
                '\r[%d/%d] B-F1: %.3f loss: %.3f | infer: %.4fs | FPS(cur): %.2f | FPS(avg): %.2f'
                % (iter, total_batches, bf1, loss.data.item(), infer_time, cur_fps, avg_fps),
                end=''
            )

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)
    scores = salEvalVal.get_scores()

    # print overall average FPS after loop
    avg_fps = (total_images / total_infer_time) if total_infer_time > 0 else 0.0
    print(f"\n[Forward-only] Total images: {total_images}, Total infer time: {total_infer_time:.4f}s, Avg FPS: {avg_fps:.2f}")

    return average_epoch_loss_val, scores, img_names, img_metrics

def ValidateSegmentation(args):
    torch.backends.cudnn.benchmark = True
    SEED = 2333
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    model = CFBNet(3, 1)

    args.savedir = args.savedir + '_' + args.file_root + '_iter_' + str(args.max_steps) + '_lr_' + str(args.lr)  + '/'
    args.vis_dir = './Predict/' + args.file_root  + '/'

    # dataset paths
    if args.file_root == 'S1G':
        args.file_root = "../S1G"
    elif args.file_root == 'etci':
        args.file_root = "./../../../data3/yxy25/etci_0306_RFANet"
    elif args.file_root == 'URBAN':
        args.file_root = '/data3/yxy25/flood/UrbanSARFloods/UrbanSARFloods_v1/testing_case_256/dataset'
    elif args.file_root == 'quick_start':
        args.file_root = './samples'
    else:
        raise TypeError('%s has not defined. Supported datasets: S1G, etci, URBAN, quick_start' % args.file_root)

    # create output directories
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    if not os.path.exists(args.vis_dir):
        os.makedirs(args.vis_dir)

    if args.onGPU:
        model = model.cuda()

    # parameter count
    total_params = sum([np.prod(p.size()) for p in model.parameters()])
    print('Total network parameters (excluding idr): ' + str(total_params))

    # normalization params
    mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]

    # test-time transforms
    valDataset = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.ToTensor()
    ])

    # load test data
    test_data = myDataLoader.Dataset("test", file_root=args.file_root, transform=valDataset)
    testLoader = torch.utils.data.DataLoader(
        test_data, shuffle=False,
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=False)

    if args.onGPU:
        cudnn.benchmark = True

    # log file setup
    logFileLoc = args.savedir + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s" % (str(total_params)))
        logger.write(
            "\n%s\t%s\t%s\t%s" % ('Epoch', 'Boundary_F1', 'Boundary_Precision', 'Boundary_Recall'))
    logger.flush()

    # load model
    model_file_name = args.savedir + 'best_model.pth'
    state_dict = torch.load(model_file_name)
    model.load_state_dict(state_dict, strict=False)

    # run inference and collect per-image metrics
    loss_test, score_test, img_names, img_metrics = val(args, testLoader, model, 0)

    # save per-image metrics to excel
    excel_data = pd.DataFrame({
        'Image': img_names,
        'Boundary_F1': [m['boundary_f1'] for m in img_metrics],
        'Boundary_Precision': [m['boundary_precision'] for m in img_metrics],
        'Boundary_Recall': [m['boundary_recall'] for m in img_metrics],
    })
    excel_path = os.path.join(args.vis_dir, 'test_cd_metrics.xlsx')
    excel_data.to_excel(excel_path, index=False, engine='openpyxl')
    print(f"\nExcel results saved to: {excel_path}")

    # print and log overall metrics
    print("\nTest :\t B-F1 (te) = %.4f\t B-Precision (te) = %.4f\t B-Recall (te) = %.4f" \
          % (score_test['Boundary_F1'], score_test['Boundary_Precision'], score_test['Boundary_Recall']))
    logger.write("\n%s\t\t%.4f\t\t%.4f\t\t%.4f" % ('Test', score_test['Boundary_F1'],
                                                    score_test['Boundary_Precision'],
                                                    score_test['Boundary_Recall']))
    logger.flush()
    logger.close()

    # save mat file
    import scipy.io as scio
    scio.savemat(args.vis_dir + 'results.mat', score_test)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file_root', default="S1G", help='Data directory | S1G | etci | URBAN | quick_start')
    parser.add_argument('--inWidth', type=int, default=256, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=256, help='Height of RGB image')
    parser.add_argument('--max_steps', type=int, default=40000, help='Max. number of iterations')
    parser.add_argument('--num_workers', type=int, default=4, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--step_loss', type=int, default=100, help='Decrease learning rate after how many epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--lr_mode', default='poly', help='Learning rate policy, step or poly')
    parser.add_argument('--savedir', default='./results', help='Directory to save the results')
    parser.add_argument('--resume', default=None, help='Use this checkpoint to continue training | '
                                                       './results_ep100/checkpoint.pth.tar')
    parser.add_argument('--logFile', default='testLog.txt',
                        help='File that stores the training and validation logs')
    parser.add_argument('--onGPU', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--weight', default='', type=str, help='pretrained weight, can be a non-strict copy')
    parser.add_argument('--ms', type=int, default=0, help='apply multi-scale training, default False')

    args = parser.parse_args()
    print('Called with args:')
    print(args)

    ValidateSegmentation(args)
