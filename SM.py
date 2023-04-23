import argparse
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from SMC_model import SMC_ViT,RMC_Classifier
from torchvision.datasets import ImageFolder
from dataset import MyCIFAR10
from transform import TwoCropsTransform,train_transform,test_transform
import numpy as np
from sklearn.metrics import confusion_matrix
import math
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os

#计算acc,ppv,sen,spec,all_acc
def Calculate_acc_ppv_sen_spec(matrix,class_num,class_names):
    results_matrix = np.zeros([class_num,4])
    #diagonal负责统计对角线元素的和
    diagonal=0
    for i in range(class_num):
        tp = matrix[i][i]
        diagonal+=tp
        fn = np.sum(matrix,axis=1)[i] - tp
        fp=np.sum(matrix,axis=0)[i]-tp
        tn=np.sum(matrix)-tp-fp-fn
        acc=(tp+tn)/(tp+tn+fp+fn)
        ppv=tp/(tp+fp)
        sen=tp/(tp+fn)
        spec=tn/(tn+fp)
        results_matrix[i][0]=acc*100
        results_matrix[i][1] = ppv * 100
        results_matrix[i][2] = sen * 100
        results_matrix[i][3] = spec * 100

    average = [0 for i in range(4)]

    for i in range(class_num):
        print('{0}：acc:{1:.2f}%,ppv:{2:.2f}%,sen:{3:.2f}%,spec:{4:.2f}%'.format(class_names[i],results_matrix[i][0],
                                                                     results_matrix[i][1],results_matrix[i][2],results_matrix[i][3]))
        average[0]+=results_matrix[i][0]
        average[1] += results_matrix[i][1]
        average[2] += results_matrix[i][2]
        average[3] += results_matrix[i][3]

    print('四项评价指标求均值,acc:{0:.2f}%,ppv:{1:.2f}%,sen:{2:.2f}%,spec:{3:.2f}%'.format(average[0]/class_num,
                                                                            average[1]/class_num, average[2]/class_num,
                                                                            average[3]/class_num))

def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr

def train():
    train_model.train()
    losses = []
    train_bar = tqdm(train_dataloader)
    for [img,_], label in train_bar:
        img = img.cuda()

        train_optimizer.zero_grad()

        predicted_img=train_model(img)
        loss = torch.mean((predicted_img - img) ** 2)

        loss.backward()
        train_optimizer.step()

        losses.append(loss.item())
        avg_loss = sum(losses) / len(losses)
        train_bar.set_description('当前训练轮数为{0},当前平均损失为:{1:.5f}'.format(epoch+1,avg_loss))
    writer.add_scalar('SM_loss',sum(losses) / len(losses),global_step=epoch+1)


def test():

    y_true = []
    y_pred = []

    correct = 0
    total = len(test_dataloader.dataset)

    test_model.eval()
    with torch.no_grad():
        batch = tqdm(test_dataloader)
        for img, label in batch:
            img = img.cuda()
            labels = label.cuda()
            outputs = test_model(img)
            _, predicted = torch.max(outputs, dim=1)

            #计算top3
            maxk = max((1, 3))
            label_resize = labels.view(-1, 1)
            _, pred = outputs.topk(maxk, 1, True, True)
            correct += torch.eq(pred, label_resize).sum().float().item()

            for i in range(len(labels)):
                label = labels[i]
                y_true.append(label.item())
                y_pred.append(predicted[i].item())

        #top1
        print('top1 acc:{0:.3f}%'.format(accuracy_score(y_true, y_pred) * 100))

        #top3
        print('top3 acc:{0:.3f}%'.format(correct / total * 100))

        writer.add_scalar('SM_top1', accuracy_score(y_true, y_pred) * 100, global_step=epoch + 1)
        writer.add_scalar('SM_top3', correct / total * 100, global_step=epoch + 1)

        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        print(cm)

        if args.test_data_path != './cifar10':
            Calculate_acc_ppv_sen_spec(cm, class_num=5, class_names=['F', 'N', 'Q', 'S', 'V'])

if __name__ == '__main__':
    #设置初试参数
    parser = argparse.ArgumentParser(description='SM')
    parser.add_argument('--train_data_path', default='./dataSetActuallyUsed/train', type=str, help='训练数据路径')
    parser.add_argument('--test_data_path', default='./dataSetActuallyUsed/test', type=str, help='测试数据路径')
    parser.add_argument('--class_num', default=5, type=int, help='几分类')
    parser.add_argument('--feature_dim', default=128, type=int, help='表征向量的长度')
    parser.add_argument('--temperature', default=0.5, type=float, help='损失函数中的temperature')
    parser.add_argument('--batch_size', default=128, type=int, help='批次中的数量')
    parser.add_argument('--num_workers', default=0, type=int, help='提前加载量')
    parser.add_argument('--epochs', default=600, type=int, help='训练轮次')
    parser.add_argument('--lr', type=float, default=1e-1,help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-6,help='权重衰减')
    parser.add_argument('--image_size', type=int, default=64, help='图片大小')
    parser.add_argument('--patch_size', type=int, default=32, help='块的大小')
    parser.add_argument('--logs_path', type=str, default='./SM_logs', help='logs地址')
    args = parser.parse_args()

    if args.train_data_path == './dataSetActuallyUsed/train':
        train_dataset = ImageFolder(args.train_data_path,transform=TwoCropsTransform(train_transform))
        test_dataset=ImageFolder(args.test_data_path,transform=test_transform)
    else:
        train_dataset = MyCIFAR10(data_path=args.train_data_path, transform=TwoCropsTransform(train_transform), train=True)
        test_dataset = MyCIFAR10(data_path=args.test_data_path, transform=test_transform, train=False)

    # print(train_dataset.class_to_idx)
    # # 没有任何的transform，所以返回的还是PIL Image对象
    # # print(dataset[0][1])# 第一维是第几张图，第二维为1返回label
    # # print(dataset[0][0]) # 为0返回图片数据
    # plt.imshow(train_dataset[0][0])
    # plt.axis('off')
    # plt.show()

    if args.train_data_path == './dataSetActuallyUsed/train':
        writer = SummaryWriter(os.path.join(args.logs_path, 'MIT-BIH'))
    else:
        writer = SummaryWriter(os.path.join(args.logs_path, 'cifar10'))

    train_dataloader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    train_model=SMC_ViT(image_size=args.image_size, patch_size=args.patch_size).cuda()
    train_optimizer = optim.SGD(train_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    test_model=RMC_Classifier(train_model.encoder, num_classes=args.class_num).cuda()
    criterion = nn.CrossEntropyLoss()
    linear_optimizer = optim.SGD(test_model.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        adjust_learning_rate(train_optimizer, args.lr, epoch, args)

        train()

        test_model.cls_token = train_model.encoder.cls_token
        test_model.pos_embedding = train_model.encoder.pos_embedding
        test_model.patchify = train_model.encoder.patchify
        test_model.transformer = train_model.encoder.transformer
        test_model.layer_norm = train_model.encoder.layer_norm

        adjust_learning_rate(linear_optimizer, args.lr, epoch, args)
        linear_train_batch=tqdm(train_dataloader)

        test_model.train()
        for [img, _], label in linear_train_batch:
            img, label = img.cuda(), label.cuda()

            linear_optimizer.zero_grad()

            logits = test_model(img)
            loss = criterion(logits, label)

            loss.backward()
            linear_optimizer.step()

            linear_train_batch.set_description('冻结主干网络参数,并着重训练后面的线性分类层')

        test()

    writer.close()

    model_save_path = './model_save'
    if not os.path.isdir(model_save_path):
        os.mkdir(model_save_path)
    torch.save(train_model.state_dict(), os.path.join(model_save_path,'SM_model'+'.pth'))