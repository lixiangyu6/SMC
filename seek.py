from torchvision import transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from PIL import Image
from SMC_model import SMC_ViT

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

model_save_path = './model_save/test_model_10.pth'
test_model=SMC_ViT(image_size=64, patch_size=32).cuda()
test_model.load_state_dict(torch.load(model_save_path))

img_file_path='./原图.png'

img = Image.open(img_file_path).convert('RGB')
img = transforms.Resize(64)(img)  # 保持长宽比的resize方法

print('展示原图')
plt.imshow(img)
plt.show()

img = transforms.ToTensor()(img)
img=img.unsqueeze(0)
img = img.cuda()

output = test_model(img)  # net是提前读取的模型
output=output.squeeze(0)
output = output.detach().cpu().numpy()
output = np.transpose(output, (1, 2, 0))  # C*H*W -> H*W*C

print('展示重构后图')
plt.imshow(output)
plt.show()

image.imsave('./output.png',output)

