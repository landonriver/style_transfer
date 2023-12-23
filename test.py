import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import PIL.Image as Image

# 检查是否有 GPU（cuda），否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载内容图像和风格图像
content_image = Image.open('./data/before.jpg')
style_image = Image.open('./data/style.jpg')

print(content_image.size)
print(style_image.size)

# 定义图像归一化的均值和标准差
rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])

# 图像预处理和后处理函数
def preprocess(img, img_shape):
    transform = transforms.Compose([
        transforms.Resize(img_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=rgb_mean, std=rgb_std),
    ])
    return transform(img).unsqueeze(0)

def postprocess(img):
    img = img[0].to(rgb_std.device)
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))

# 加载预训练的 VGG19 模型
model = torchvision.models.vgg19(pretrained=True)

# 指定用于风格和内容提取的层
style_layers = [0, 5, 10, 19, 28]
content_layers = [25]

# 使用 VGG19 提取特征
net = nn.Sequential(*[model.features[i] for i in range(max(content_layers + style_layers) + 1)])

def extract_features(X, content_layers, style_layers):
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles

# 从内容图像获取内容特征
def get_contents(image_shape, device):
    content_X = preprocess(content_image, image_shape).to(device)
    content_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, content_Y

# 从风格图像获取风格特征
def get_styles(image_shape, device):
    style_X = preprocess(style_image, image_shape).to(device)
    _, style_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, style_Y

# 内容损失函数
def content_loss(Y_hat, Y):
    return torch.square(Y_hat - Y.detach()).mean()

# 风格损失函数中的 Gram 矩阵计算
def gram(X):
    num_channels, n = X.shape[1], X.numel() // X.shape[1]
    X = X.reshape((num_channels, n))
    return torch.matmul(X, X.T) / (num_channels * n)

# 风格损失函数
def style_loss(Y_hat, gram_Y):
    return torch.square(gram(Y_hat) - gram_Y.detach()).mean()

# 总变差损失函数
def tv_loss(Y_hat):
    tv_loss = 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                     torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())
    return tv_loss

# 内容、风格和总变差损失的权重
content_weight, style_weight, tv_weight = 1, 1e4, 10

# 计算总损失
def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    content_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(contents_Y_hat, contents_Y)]
    style_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    l = sum(content_l + style_l + [tv_l])
    return content_l, style_l, tv_l, l

# 生成合成图像的模型
class SynthesizedImage(nn.Module):
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weight

# 获取合成图像的初始值、风格 Gram 矩阵和优化器
def get_inits(X, device, lr, style_Y):
    gen_img = SynthesizedImage(X.shape).to(device)
    gen_img.weight.data.copy_(X.data)
    optimizer = torch.optim.Adam(gen_img.parameters(), lr)
    style_Y_gram = [gram(Y) for Y in style_Y]
    return gen_img(), style_Y_gram, optimizer

# 训练函数
def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    X, styles_Y_gram, optimizer = get_inits(X, device, lr, styles_Y)
    schedule = torch.optim.lr_scheduler.StepLR(optimizer, lr_decay_epoch, 0.8)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        content_Y_hat, style_Y_hat = extract_features(X, content_layers, style_layers)

        contents_l, styles_l, tv_l, l = compute_loss(X, content_Y_hat, style_Y_hat, contents_Y, styles_Y_gram)

        l.backward()
        optimizer.step()
        schedule.step()
        if (epoch + 1) % 5 == 0:
            print("迭代次数：{}   内容损失：{:.9f}  风格损失：{:.9f}  总变差损失：{:.9f}".format(
                epoch + 1, sum(contents_l).item(), sum(styles_l).item(), tv_l.item()
            ))
    return X

# 设置图像形状和设备，并将网络移至指定设备
device, image_shape = device, (300, 450)
net = net.to(device)

# 从内容图像获取内容特征
content_X, contents_Y = get_contents(image_shape, device)

# 从风格图像获取风格特征
_, style_Y = get_styles(image_shape, device)

# 训练模型50个迭代并显示结果
output = train(content_X, contents_Y, style_Y, device, 0.3, 50, 50)
output = postprocess(output)

# 显示合成图像
plt.imshow(output)
plt.show()

# 训练模型额外300个迭代并显示结果
output = train(content_X, contents_Y, style_Y, device, 0.3, 300, 50)
output = postprocess(output)

# 显示合成图像
plt.imshow(output)
plt.show()
