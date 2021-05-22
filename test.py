import argparse
from pathlib import Path

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import IMAGE_Dataset

CUDA_DEVICES = 0


def sortSecond(val):
    return val[1]


def test(args):
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])
    ])
    test_set = IMAGE_Dataset(Path(args.img_dir), data_transform)
    data_loader = DataLoader(
        dataset=test_set, batch_size=32, shuffle=False, num_workers=1)
    classes = [_dir.name for _dir in Path(args.img_dir).glob('*')]

    # load model
    model = torch.load(args.weights)
    model = model.cuda(CUDA_DEVICES)
    model.eval()

    total_correct = 0
    total = 0
    f1_score = 0.0
    class_correct = list(0. for i in enumerate(classes))
    class_total = list(0. for i in enumerate(classes))

    label_num = torch.zeros((1, args.class_num))
    predict_num = torch.zeros((1, args.class_num))
    acc_num = torch.zeros((1, args.class_num))

    # csv
    import csv
    with open('agriculture.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(['ID', 'LABEL'])

    with torch.no_grad():
        for inputs, labels, paths in data_loader:
            inputs = Variable(inputs.cuda(CUDA_DEVICES))
            labels = Variable(labels.cuda(CUDA_DEVICES))
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # print(paths)
            # print(predicted)
            # print(labels)

            # total
            total += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()

            # print(predict_class_id)
            # batch size
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

                p_num = int(predicted[i])
                l_num = int(labels[i])

                # print gt
                # print('{}\t{}\t{}'.format(paths[i], p_num, l_num))
                with open('agriculture.csv', 'a', newline='') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow([paths[i], p_num])

            pre_mask = torch.zeros(outputs.size()).scatter_(1, predicted.cpu().view(-1, 1), 1.)
            predict_num += pre_mask.sum(0)
            tar_mask = torch.zeros(outputs.size()).scatter_(1, labels.data.cpu().view(-1, 1), 1.)
            label_num += tar_mask.sum(0)
            acc_mask = pre_mask * tar_mask
            acc_num += acc_mask.sum(0)
        #            print('------------------')
        #            print(predict_num)
        #            print(label_num)
        #            print(acc_num)

        recall = acc_num / label_num
        precision = acc_num / predict_num
        F1 = 2 * recall * precision / (recall + precision)
        accuracy = acc_num.sum(1) / label_num.sum(1)

    F1_num = F1.numpy()
    for i, c in enumerate(classes):
        print('Accuracy of %5s : %8.4f %%' % (
            c, 100 * class_correct[i] / class_total[i]), end='')

        if np.isnan(F1_num[0][i]):
            F1_num[0][i] = 0
        print(' , f1-score of %5s : %8.4f %%' % (
            c, 100 * F1_num[0][i]))

    # Accuracy
    print('\nAccuracy on the ALL test images: %.4f %%'
          % (100 * total_correct / total))

    # f1-score
    f1_score = 100 * (F1.sum()) / args.class_num
    print('Total f1-score : %4f %%' % (f1_score))
    csvFile.close()


# Arguments
parser = argparse.ArgumentParser(description="2019 CCU AI Hackathon")
parser.add_argument('-c', '--class_num', type=int, help='number of classes', required=True)
parser.add_argument('-i', '--img_dir', type=str, help='training images folder', required=True)
parser.add_argument('-w', '--weights', type=str, help='input weights', required=True)
args = parser.parse_args()

if __name__ == '__main__':
    print('-'*64)
    test(args)
