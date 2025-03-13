import os, argparse, time, datetime, stat, shutil
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from util.RGB_Depth_dataset import LunarSeg_dataset,NPO_dataset
from util.augmentation import RandomFlip, RandomCrop, RandomCropOut, RandomBrightness, RandomNoise
from util.util import compute_results,LovaszSoftmaxLoss
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from model.LuSeg import LuSeg_S, LuSeg_RGB, LuSeg_Depth, NTXentLoss
import sys

#############################################################################################
parser = argparse.ArgumentParser(description='Train with pytorch')
parser.add_argument('--model_name', '-m', type=str, default='LuSeg')
parser.add_argument('--batch_size', '-b', type=int, default=4)
parser.add_argument('--gpu', '-g', type=int, default=0)
parser.add_argument('--need_m', '-need_m', type=int, default=7)
parser.add_argument('--sleep', '-sleep', type=int, default=1)
parser.add_argument('--lr_start', '-ls', type=float, default=0.01)
parser.add_argument('--lr_decay', '-ld', type=float, default=0.95)
parser.add_argument('--epoch_max', '-em', type=int, default=200)  # please stop training mannully
parser.add_argument('--epoch_from', '-ef', type=int, default=0)
parser.add_argument('--num_workers', '-j', type=int, default=8)
parser.add_argument('--n_class', '-nc', type=int, default=3)
parser.add_argument('--data_dir', '-dr', type=str, default='/data/shuaifeng/LunarSeg')
parser.add_argument('--data_name', '-dn', type=str, default='LunarSeg')
args = parser.parse_args()
#############################################################################################

augmentation_methods = [
    RandomFlip(prob=0.5),
    RandomCrop(crop_rate=0.1, prob=1.0),
    # RandomCropOut(crop_rate=0.2, prob=1.0),
    # RandomBrightness(bright_range=0.15, prob=0.9),
    # RandomNoise(noise_range=5, prob=0.9),
]


def train(epo, model_s, train_loader, optimizer,criterion):
    model_s.train()
    for it, (images, labels, names) in enumerate(train_loader):
        images = Variable(images).cuda(args.gpu)
        labels = Variable(labels).cuda(args.gpu)
        start_t = time.time()  # time.time() returns the current time
        optimizer.zero_grad()
        rgb, rgb_result = model_s(images)  # depth image result, rgb image result, rgb image pre-segmentaion,  complement feature

        loss_r = criterion(rgb_result,labels)  # Note that the cross_entropy function has already include the softmax function

        loss = loss_r

        loss.backward()
        optimizer.step()
        lr_this_epo = 0
        for param_group in optimizer.param_groups:
            lr_this_epo = param_group['lr']
        print(
            'Train: %s, epo %s/%s, iter %s/%s, lr %.8f, %.2f img/sec, loss %.4f, time %s' \
            % (args.model_name, epo, args.epoch_max, it + 1, len(train_loader), lr_this_epo,
               len(names) / (time.time() - start_t), float(loss), datetime.datetime.now().replace(microsecond=0) - start_datetime))
        if accIter['train'] % 1 == 0:

            writer.add_scalar('Train/loss_r', loss_r, accIter['train'])

        view_figure = True  # note that I have not colorized the GT and predictions here
        if accIter['train'] % 10 == 0:
            if view_figure:
                input_rgb_images = vutils.make_grid(images[:, :3], nrow=8,
                                                    padding=10)  # can only display 3-channel images, so images[:,:3]
                writer.add_image('Train/input_rgb_images', input_rgb_images, accIter['train'])
                scale = max(1,
                            255 // args.n_class)  # label (0,1,2..) is invisable, multiply a constant for visualization
                groundtruth_tensor = labels.unsqueeze(1) * scale  # mini_batch*480*640 -> mini_batch*1*480*640
                groundtruth_tensor = torch.cat((groundtruth_tensor, groundtruth_tensor, groundtruth_tensor),
                                               1)  # change to 3-channel for visualization
                groudtruth_images = vutils.make_grid(groundtruth_tensor, nrow=8, padding=10)
                writer.add_image('Train/groudtruth_images', groudtruth_images, accIter['train'])

                predicted_rgb = rgb_result.argmax(1).unsqueeze(
                    1) * scale  # mini_batch*args.n_class*480*640 -> mini_batch*480*640 -> mini_batch*1*480*640
                predicted_rgb = torch.cat((predicted_rgb, predicted_rgb, predicted_rgb),
                                          1)  # change to 3-channel for visualization, mini_batch*1*480*640
                predicted_rgb_images = vutils.make_grid(predicted_rgb, nrow=8, padding=10)
                writer.add_image('Train/predicted_rgb_images', predicted_rgb_images, accIter['train'])


                depth_tensor = images[:,
                               3:4]  # mini_batch*args.n_class*480*640 -> mini_batch*480*640 -> mini_batch*1*480*640
                depth_tensor = torch.cat((depth_tensor, depth_tensor, depth_tensor),
                                         1)  # change to 3-channel for visualization, mini_batch*1*480*640
                depth_tensor = vutils.make_grid(depth_tensor, nrow=8, padding=10)
                writer.add_image('Train/depth_tensor', depth_tensor, accIter['train'])
        accIter['train'] = accIter['train'] + 1


def validation(epo, model, val_loader,criterion):
    model.eval()
    with torch.no_grad():
        for it, (images, labels, names) in enumerate(val_loader):
            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            start_t = time.time()  # time.time() returns the current time

            _, rgb_result = model(images)
            loss_r = criterion(rgb_result,labels)  # Note that the cross_entropy function has already include the softmax function

            loss = loss_r

            print('Val: %s, epo %s/%s, iter %s/%s, %.2f img/sec, loss %.4f, time %s' \
                  % (
                  args.model_name, epo, args.epoch_max, it + 1, len(val_loader), len(names) / (time.time() - start_t),
                  float(loss),
                  datetime.datetime.now().replace(microsecond=0) - start_datetime))
            if accIter['val'] % 1 == 0:
                writer.add_scalar('Validation/loss', loss, accIter['val'])
                writer.add_scalar('Validation/loss_r', loss_r, accIter['val'])



def testing(epo, model, test_loader):
    model.eval()
    conf_total = np.zeros((args.n_class, args.n_class))
    label_list = ["background", "negative", "positive"]
    testing_results_file = os.path.join(weight_dir, 'testing_results_file.txt')
    with torch.no_grad():
        for it, (images, labels, names) in enumerate(test_loader):
            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            _, rgb_result = model(
                images)  # depth image result, rgb image result, rgb image pre-segmentaion,  complement feature
            logits = rgb_result
            label = labels.cpu().numpy().squeeze().flatten()
            prediction = logits.argmax(
                1).cpu().numpy().squeeze().flatten()  # prediction and label are both 1-d array, size: minibatch*640*480
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0, 1,
                                                                             2])  # conf is args.n_class*args.n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
            conf_total += conf
            print('Test: %s, epo %s/%s, iter %s/%s, time %s' % (
            args.model_name, epo, args.epoch_max, it + 1, len(test_loader),
            datetime.datetime.now().replace(microsecond=0) - start_datetime))
    precision, recall, IoU, F1 = compute_results(conf_total)
    writer.add_scalar('Test/average_precision', precision.mean(), epo)
    writer.add_scalar('Test/average_recall', recall.mean(), epo)
    writer.add_scalar('Test/average_IoU', IoU.mean(), epo)
    writer.add_scalar('Test/average_F1', F1.mean(), epo)
    for i in range(len(precision)):
        writer.add_scalar("Test(class)/precision_class_%s" % label_list[i], precision[i], epo)
        writer.add_scalar("Test(class)/recall_class_%s" % label_list[i], recall[i], epo)
        writer.add_scalar('Test(class)/Iou_%s' % label_list[i], IoU[i], epo)
        writer.add_scalar('Test(class)/F1_%s' % label_list[i], F1[i], epo)
    if epo == 0:
        with open(testing_results_file, 'w') as f:
            f.write("# %s, initial lr: %s, batch size: %s, date: %s \n" % (
            args.model_name, args.lr_start, args.batch_size, datetime.date.today()))
            f.write("# epoch: background, negative, positive, average(nan_to_num). (Acc %, IoU %)\n")
    with open(testing_results_file, 'a') as f:
        f.write(str(epo) + ': ')
        for i in range(len(precision)):
            f.write('%0.4f,%0.4f,%0.4f,%0.4f, ' % (100 * precision[i], 100 * recall[i], 100 * IoU[i], 100 * F1[i]))
        f.write('%0.4f,%0.4f,%0.4f,%0.4f\n' % (
        100 * np.mean(np.nan_to_num(precision)), 100 * np.mean(np.nan_to_num(recall)),
        100 * np.mean(np.nan_to_num(IoU)), 100 * np.mean(np.nan_to_num(F1))))
    print('saving testing results.')
    with open(testing_results_file, "r") as file:
        writer.add_text('testing_results', file.read().replace('\n', '  \n'), epo)


if __name__ == '__main__':

    torch.cuda.set_device(args.gpu)
    print("\nthe pytorch version:", torch.__version__)
    print("the gpu count:", torch.cuda.device_count())
    print("the current used gpu:", torch.cuda.current_device(), '\n')

    model_s = LuSeg_RGB(n_class=args.n_class)

    if args.gpu >= 0:
        model_s.cuda(args.gpu)
    optimizer = torch.optim.SGD(model_s.parameters(), lr=args.lr_start, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay, last_epoch=-1)

    # preparing folders
    if os.path.exists("./runs_RGB"):
        shutil.rmtree("./runs_RGB")
    weight_dir = os.path.join("./runs_RGB", args.model_name)
    os.makedirs(weight_dir)

    writer = SummaryWriter("./runs_RGB/tensorboard_log")

    print('training %s on GPU #%d with pytorch' % (args.model_name, args.gpu))
    print('from epoch %d / %s' % (args.epoch_from, args.epoch_max))
    print('weight will be saved in: %s' % weight_dir)
    
    if args.data_name == 'LunarSeg':
        train_dataset = LunarSeg_dataset(data_dir=args.data_dir, split='train', transform=augmentation_methods)
        val_dataset = LunarSeg_dataset(data_dir=args.data_dir, split='validation')
        test_dataset = LunarSeg_dataset(data_dir=args.data_dir, split='test')
    elif args.data_name == 'NPO':
        train_dataset = NPO_dataset(data_dir=args.data_dir, split='train', transform=augmentation_methods)
        val_dataset = NPO_dataset(data_dir=args.data_dir, split='validation')
        test_dataset = NPO_dataset(data_dir=args.data_dir, split='test')
    else:
        print("please check your dataset name!")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    contrastive_loss_fn = NTXentLoss(temperature=0.5)
    criterion = LovaszSoftmaxLoss()
    start_datetime = datetime.datetime.now().replace(microsecond=0)
    accIter = {'train': 0, 'val': 0}
    for epo in range(args.epoch_from, args.epoch_max):
        print('\ntrain %s, epo #%s begin...' % (args.model_name, epo))
        # scheduler.step() # if using pytorch 0.4.1, please put this statement here
        train(epo, model_s, train_loader, optimizer,criterion)
        validation(epo, model_s, val_loader,criterion)

        checkpoint_model_file = os.path.join(weight_dir, str(epo) + '.pth')
        print('saving check point %s: ' % checkpoint_model_file)
        torch.save(model_s.state_dict(), checkpoint_model_file)

        testing(epo, model_s,test_loader)  # testing is just for your reference, you can comment this line during training
        scheduler.step()  # if using pytorch 1.1 or above, please put this statement here
