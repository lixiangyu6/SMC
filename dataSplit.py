import os
import random
from shutil import copy2

def dataset_split(source, destination, train_scale=0.9):
    '''
    读取源数据文件夹，生成划分好的文件夹，分为trian、test
    :param source: 源文件夹
    :param destination: 目标文件夹
    :param train_scale: 训练集比例
    '''
    print("开始数据集划分")
    class_names = os.listdir(source)
    #如果目标文件夹不存在则创建
    if not os.path.isdir(destination):
        os.mkdir(destination)
    # 在目标目录下创建文件夹
    split_names = ['train','test']
    for split_name in split_names:
        split_path = os.path.join(destination, split_name)
        if os.path.isdir(split_path):
            pass
        else:
            os.mkdir(split_path)
        # 然后在split_path的目录下创建类别文件夹
        for class_name in class_names:
            class_split_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_split_path):
                pass
            else:
                os.mkdir(class_split_path)

    # 按照比例划分数据集，并进行数据图片的复制
    # 首先进行分类遍历
    for class_name in class_names:
        current_data_dir = os.path.join(source, class_name)
        current_all_data = os.listdir(current_data_dir)
        current_data_length = len(current_all_data)
        current_data_index_list = list(range(current_data_length))
        random.shuffle(current_data_index_list)

        train_folder = os.path.join(os.path.join(destination, 'train'), class_name)
        test_folder = os.path.join(os.path.join(destination, 'test'), class_name)
        train_stop_flag = int(current_data_length * train_scale)
        train_num = 0
        test_num = 0

        for i in current_data_index_list:
            src_img_path = os.path.join(current_data_dir, current_all_data[i])
            if train_num+1 <= train_stop_flag:
                copy2(src_img_path, train_folder)
                train_num = train_num + 1
            else:
                copy2(src_img_path, test_folder)
                test_num = test_num + 1

        print("{0}".format(class_name))
        print("训练集{0}：{1}张".format(train_folder, train_num))
        print("测试集{0}：{1}张".format(test_folder, test_num))

if __name__ == '__main__':
    dataset_split('./GramianAngularField_data','./dataSetActuallyUsed')