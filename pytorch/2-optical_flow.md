#### 1.光流（optical flow）

光流的任务是预测两个图像之间的运动，通常是两个连续的视频帧。光流模型以两幅图像作为输入，预测结果是一个流，流表示第一幅图像中每个像素的位移，并映射到第二幅图像中对应的像素。流用一个维度为（N，2，H，w）的矩阵表示，其中数字2对应于预测的水平和垂直位移，N表示batch的大小。

#### 2.用torchvision官方实现的RAFT模型进行预测

```python
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import functional as F
from torchvision import transforms as T
from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image
from torchvision.io import read_video, write_jpeg

# 以grid形式展示图片
def plot(imgs, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])

    for row, img_batch in enumerate(imgs):
        for col, img in enumerate(img_batch, 1):
            plt.subplot(num_rows, num_cols, row * num_cols + col)
            img = img.permute(1, 2, 0)
            plt.imshow(img)
    plt.show()

# 图片数据预处理，类型转换（将0-255间的tensor转换成0.0-1.0间的图像tensor），标准化，调整尺寸
def preprocess(batch):
    transforms = T.Compose([
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=0.5, std=0.5),
        T.Resize(size=(520, 960))
    ])
    batch = transforms(batch)
    return batch


vframs, aframs, info = read_video("D:/tmp/basketball.mp4", pts_unit='sec') # 读视频，返回视频帧，音频帧，元数据
frames = vframs.permute(0, 3, 1, 2) # 视频帧的形状是（帧数，H，W，C），转换成（帧数，C，H，W）
img1_batch = torch.stack([frames[100], frames[150]]) # 将第100帧和150帧的图片合成一个batch，作为模型预测的前帧
img2_batch = torch.stack([frames[101], frames[151]]) # 将第101帧和151帧的图片合成一个batch，作为模型预测的后帧
plot(img1_batch) # 展示前帧的图片
plot(img2_batch)

device = "cpu"
img1_batch = preprocess(img1_batch).to(device) # 预处理数据
img2_batch = preprocess(img2_batch).to(device)
print(f"shape = {img1_batch.shape}, dtype = {img1_batch.dtype}")

model = raft_large(progress=False).to(device) # 实例化光流模型
model_weight = torch.load("D:/pretrained_weights/optical_flow/raft_large.pth") # 读取权重
model.load_state_dict(model_weight) # 加载权重
model = model.eval() # 设置为评估模式，此时dropout不生效，BN采用训练集的样本分布
list_of_flows = model(img1_batch.to(device), img2_batch.to(device)) # 预测，模型返回光流
print(f"type = {type(list_of_flows)}")
print(f"length = {len(list_of_flows)} = number of iterations of the model")
predicted_flows = list_of_flows[-1] # 取最后一个光流作为预测结果，结果形状为（批大小，2，H，W），表示图像上像素点在水平和垂直方向上的位移
print(f"dtype = {predicted_flows.dtype}")
print(f"shape = {predicted_flows.shape} = (N, 2, H, W)")
print(f"min = {predicted_flows.min()}, max = {predicted_flows.max()}")

flow_imgs = flow_to_image(predicted_flows) # 将预测结果转换成图片，此时图片数值为-1.0-1.0之间
img1_batch = [(img1 + 1) / 2 for img1 in img1_batch] # 将图片数值转换为0.0-1.0之间
grid = [[img1, flow_img] for (img1, flow_img) in zip(img1_batch, flow_imgs)] # 将前帧图片和预测结果组成grid，并画出
plot(grid)


for i, (img1, img2) in enumerate(zip(frames, frames[1:])): # 将整部视频拆分成多个前帧和后帧
    img1 = preprocess(img1[None]).to(device) # img1[None]会给img1添加一个维度在最前面
    img2 = preprocess(img2[None]).to(device)
	
    list_of_flows = model(img1, img2)
    predicted_flow = list_of_flows[-1][0] # 取出预测结果batch中的第一个
    flow_img = flow_to_image(predicted_flow).to("cpu") # 将预测结果转换成图片
    output_folder = "D:/tmp/"  # Update this to the folder of your choice
    write_jpeg(flow_img, output_folder + f"predicted_flow_{i}.jpg") # 保存图片
    
# 可以在命令行输入如下ffmpeg的命令，将预测的各个光流结果合成一个动态的gif图
ffmpeg -f image2 -framerate 30 -i predicted_flow_%d.jpg -loop -1 flow.gif

```

