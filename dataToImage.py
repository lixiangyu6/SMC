import numpy as np
import os
from matplotlib import image
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField

if __name__=='__main__':
    all_x = np.load(os.path.join('./', 'data_save', 'AllData.npy'))
    all_y = np.load(os.path.join('./', 'data_save', 'AllLabel.npy'))

    categories=['N','S','V','F','Q']
    categoryWithNum={'N':0,'S':0,'V':0,'F':0,'Q':0}

    #创建各个目录
    main_path = './GramianAngularField_data'
    if not os.path.isdir(main_path):
        os.mkdir(main_path)
    for item in categories:
        eachCategory=os.path.join(main_path,item)
        if not os.path.isdir(eachCategory):
            os.mkdir(eachCategory)

    for index,x in enumerate(all_x):
        x=x.reshape(1,-1)
        label=int(all_y[index])

        image_size = 64  #生成的 GAF 图片的大小

        # `method` 的可选参数有：`summation` and `difference`
        # The optional parameters of argument `method`: `summation` and `difference`
        gasf = GramianAngularField(image_size=image_size, method='summation')
        sin_gasf = gasf.fit_transform(x)

        # gadf = GramianAngularField(image_size=image_size, method='difference')
        # sin_gadf = gadf.fit_transform(x)
        #
        # imges = [sin_gasf[0], sin_gadf[0]]
        # titles = ['Summation', 'Difference']

        # # 两种方法的可视化差异对比
        # # Comparison of two different methods
        # fig, axs = plt.subplots(1, 2, constrained_layout=True)
        # for img, title, ax in zip(imges, titles, axs):
        #     ax.imshow(img)
        #     ax.set_title(title)
        # fig.suptitle('GramianAngularField', y=0.94, fontsize=16)
        # plt.margins(0, 0)
        # plt.savefig("./GramianAngularField.pdf", pad_inches=0)
        # plt.show()

        categoryWithNum[categories[label]]+=1
        pngName=str(categoryWithNum[categories[label]])+'.png'
        savePath=os.path.join(main_path,categories[label],pngName)
        image.imsave(savePath, sin_gasf[0])

        print('完成了第{0}张图片的存储，该图片的类别为:{1}'.format(index+1,categories[label]))

    print('全部图片存储完毕,各个类别图片数量为:')
    print(categoryWithNum)


