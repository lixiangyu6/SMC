from dataset import MyImageFolder,MyCIFAR10_PCL
import argparse
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from SMC_model import SMC_ViT,Dim_reduction_net,RMC_Classifier
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
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

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

def compute_features(eval_loader, model, dim_net, args):

    model.eval()
    dim_net.eval()

    features = torch.zeros(len(eval_loader.dataset), args.feature_dim).cuda()
    i=0
    batch=tqdm(eval_loader)
    for [out,_],_,_ in batch:
        with torch.no_grad():
            out = out.cuda(non_blocking=True)
            out = model(out)
            out=dim_net(out)
            for out_single in out:
                features[i]=out_single
                i+=1

        batch.set_description('计算表征向量')

    model.train()
    return features.cpu()

def run_clustering(x, args):
    results = {'im2cluster': [], 'centroids': [], 'density': []}
    batch=tqdm(args.num_cluster)
    for num_cluster in batch:
        gmm = GaussianMixture(n_components=int(num_cluster)).fit(x)
        im2cluster = gmm.predict(x)
        # im2cluster=KMeans(n_clusters=int(num_cluster)).fit_predict(x)
        im2cluster=[int(i) for i in im2cluster]
        points_to_classes=[[] for i in range(int(num_cluster))]
        centroids=[]

        for i,x_single in enumerate(x):
            label=im2cluster[i]
            points_to_classes[label].append(x_single)

        for points in points_to_classes:
            if len(points)==1:
                centroids.append(points[0])
            else:
                centroids.append(sum(points)/len(points))
        centroids=np.asarray(centroids)

        # sample-to-centroid distances for each cluster
        Dcluster = [[] for c in range(int(num_cluster))]
        for im, i in enumerate(im2cluster):
            Dcluster[i].append(np.linalg.norm(x[im]-centroids[i]))

        # concentration estimation (phi)
        density = np.zeros(int(num_cluster))
        for i, dist in enumerate(Dcluster):
            if len(dist) > 1:
                d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)
                density[i] = d

                # if cluster only has one point, use the max to estimate its concentration
        dmax = density.max()
        for i, dist in enumerate(Dcluster):
            if len(dist) <= 1:
                density[i] = dmax

        density = density.clip(np.percentile(density, 10),
                               np.percentile(density, 90))  # clamp extreme values for stability
        density = args.temperature * density / density.mean()  # scale the mean to temperature

        centroids = torch.Tensor(centroids).cuda()
        centroids = nn.functional.normalize(centroids, p=2, dim=1)
        im2cluster = torch.LongTensor(im2cluster).cuda()
        density = torch.Tensor(density).cuda()

        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)

        batch.set_description('进行聚类计算')

    return results

def train():

    train_model.train()
    losses = []
    train_bar = tqdm(train_dataloader)

    for i, ([img,_], index,_) in enumerate(train_bar):
        img = img.cuda()

        train_optimizer.zero_grad()

        predicted_img=train_model(img)
        q=dim_reduction_net(predicted_img)
        loss = torch.mean((predicted_img - img) ** 2)

        if epoch>=args.warm_epochs:
            proto_labels = []
            proto_logits = []
            for n, (im2cluster, prototypes, density) in enumerate(zip(cluster_result['im2cluster'], cluster_result['centroids'], cluster_result['density'])):
                # get positive prototypes
                pos_proto_id = im2cluster[index]
                pos_prototypes = prototypes[pos_proto_id]

                # sample negative prototypes
                all_proto_id = [i for i in range(im2cluster.max())]
                neg_proto_id = set(all_proto_id) - set(pos_proto_id.tolist())
                # neg_proto_id = sample(neg_proto_id, args.m)  # sample r negative prototypes
                neg_prototypes = prototypes[list(neg_proto_id)]

                proto_selected = torch.cat([pos_prototypes, neg_prototypes], dim=0)


                # compute prototypical logits
                logits_proto = torch.mm(q, proto_selected.t())

                # targets for prototype assignment
                labels_proto = torch.linspace(0, q.size(0) - 1, steps=q.size(0)).long().cuda()


                # scaling temperatures for the selected prototypes
                temp_proto = density[torch.cat([pos_proto_id, torch.LongTensor(list(neg_proto_id)).cuda()], dim=0)]
                logits_proto /= temp_proto

                proto_labels.append(labels_proto)
                proto_logits.append(logits_proto)

            loss_proto = 0
            for proto_out, proto_target in zip(proto_logits, proto_labels):
                loss_proto += criterion(proto_out, proto_target)
            # average loss across all sets of prototypes
            loss_proto /= len(args.num_cluster)
            loss += loss_proto

        loss.backward()
        train_optimizer.step()

        # 动量法更新参数
        for parameter_q, parameter_k in zip(train_model.parameters(), associated_model.parameters()):
            parameter_k.data.copy_(parameter_k.data * args.momentum + parameter_q.data * (1.0 - args.momentum))

        losses.append(loss.item())
        avg_loss = sum(losses) / len(losses)
        train_bar.set_description('当前训练轮数为{0},当前平均损失为:{1:.5f}'.format(epoch+1,avg_loss))
    writer.add_scalar('SMC_loss',sum(losses) / len(losses),global_step=epoch+1)


def test():

    y_true = []
    y_pred = []

    correct = 0
    total = len(test_dataloader.dataset)

    test_model.eval()
    with torch.no_grad():
        batch = tqdm(test_dataloader)
        for img,_,label in batch:
            img = img.cuda()
            labels = label.cuda()
            outputs = test_model(img)
            _, predicted = torch.max(outputs, dim=1)

            # 计算top3
            maxk = max((1, 3))
            label_resize = labels.view(-1, 1)
            _, pred = outputs.topk(maxk, 1, True, True)
            correct += torch.eq(pred, label_resize).sum().float().item()

            for i in range(len(labels)):
                label = labels[i]
                y_true.append(label.item())
                y_pred.append(predicted[i].item())

        # top1
        print('top1 acc:{0:.3f}%'.format(accuracy_score(y_true, y_pred) * 100))

        # top3
        print('top3 acc:{0:.3f}%'.format(correct / total * 100))

        writer.add_scalar('SMC_top1', accuracy_score(y_true, y_pred) * 100, global_step=epoch + 1)
        writer.add_scalar('SMC_top3', correct / total * 100, global_step=epoch + 1)

        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        print(cm)

        if args.test_data_path != './cifar10':
            Calculate_acc_ppv_sen_spec(cm, class_num=5, class_names=['F', 'N', 'Q', 'S', 'V'])

if __name__ == '__main__':
    #设置初试参数
    parser = argparse.ArgumentParser(description='SMC')
    parser.add_argument('--train_data_path', default='./dataSetActuallyUsed/train', type=str, help='训练数据路径')
    parser.add_argument('--test_data_path', default='./dataSetActuallyUsed/test', type=str, help='测试数据路径')
    parser.add_argument('--class_num', default=5, type=int, help='几分类')
    parser.add_argument('--num-cluster', default='5', type=str, help='聚类数量')
    parser.add_argument('--image_size', type=int, default=64, help='图片大小')
    parser.add_argument('--patch_size', type=int, default=32, help='块的大小')
    parser.add_argument('--feature_dim', default=128, type=int, help='表征向量的长度')
    parser.add_argument('--temperature', default=0.5, type=float, help='损失函数中的temperature')
    parser.add_argument('--batch_size', default=128, type=int, help='批次中的数量')
    parser.add_argument('--num_workers', default=0, type=int, help='提前加载量')
    parser.add_argument('--epochs', default=600, type=int, help='训练轮次')
    parser.add_argument('--warm_epochs', default=10, type=int, help='从何时开始加入聚类方法')
    parser.add_argument('--lr', type=float, default=1e-1,help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-6,help='权重衰减')
    parser.add_argument('--logs_path', type=str, default='./SMC_logs', help='logs地址')
    parser.add_argument('--momentum', default=0.999, type=float, help='动量参数')
    args = parser.parse_args()
    args.num_cluster = args.num_cluster.split(',')

    if args.train_data_path == './dataSetActuallyUsed/train':
        train_dataset = MyImageFolder(args.train_data_path,transform=TwoCropsTransform(train_transform))
        cluster_dataset = MyImageFolder(args.train_data_path, transform=test_transform)
        test_dataset=MyImageFolder(args.test_data_path,transform=test_transform)
    else:
        train_dataset = MyCIFAR10_PCL(data_path=args.train_data_path, transform=TwoCropsTransform(train_transform),train=True)
        cluster_dataset = MyCIFAR10_PCL(args.train_data_path, transform=test_transform, train=True)
        test_dataset = MyCIFAR10_PCL(data_path=args.test_data_path, transform=test_transform, train=False)

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

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    train_model=SMC_ViT(image_size=args.image_size, patch_size=args.patch_size).cuda()
    associated_model=SMC_ViT(image_size=args.image_size, patch_size=args.patch_size).cuda()#伴生网络
    test_model=RMC_Classifier(train_model.encoder, num_classes=args.class_num).cuda()#测试网络
    dim_reduction_net=Dim_reduction_net(image_size=args.image_size,vectordim=args.feature_dim).cuda()#降维网络

    for param_q, param_k in zip(train_model.parameters(), associated_model.parameters()):#复制参数
        param_k.data.copy_(param_q.data)
        param_k.requires_grad=False

    for param_q in dim_reduction_net.parameters():#该网络仅做降维用，因此不需要更新参数
        param_q.requires_grad=False

    train_optimizer = optim.SGD(train_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    linear_optimizer = optim.SGD(test_model.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):

        cluster_result = None
        if epoch >= args.warm_epochs:
            # compute momentum features for center-cropped images
            features = compute_features(train_dataloader, associated_model, dim_reduction_net,args)
            features[torch.norm(features, dim=1) > 1.5] /= 2  # account for the few samples that are computed twice
            features = features.numpy()
            cluster_result = run_clustering(features, args)

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
        for [img, _],_,label in linear_train_batch:
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
    torch.save(train_model.state_dict(), os.path.join(model_save_path, 'SMC_model' + '.pth'))