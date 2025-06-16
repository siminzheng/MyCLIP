import torch
from torch.utils.data import DataLoader
from data.dataset import load_test_dataset
from models.clip_model import CLIP
from utils.tokenizer_utils import tokenizer, tokenize_texts

# 设备选择，优先使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate():
    # 加载测试数据集和类别名称（class_names是类别的字符串列表）
    test_dataset, class_names = load_test_dataset()
    # DataLoader用于批量加载测试数据，batch_size=1，不打乱顺序
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 取词表大小，初始化模型时需要
    vocab_size = tokenizer.vocab_size
    # 初始化CLIP模型，移动到设备（GPU或CPU）
    model = CLIP(vocab_size).to(device)

    # 加载训练好的模型权重（第10个epoch保存的）
    ckpt_path = "checkpoints/clip_epoch_10.pt"
    state_dict = torch.load(ckpt_path, map_location=device)  # 根据设备加载
    model.load_state_dict(state_dict)
    model.eval()  # 切换到评估模式，关闭dropout等

    # 将类别文本转换成token id张量，放到设备上
    text_tokens = tokenize_texts(class_names).to(device)
    with torch.no_grad():  # 评估时不需要计算梯度
        # 计算文本特征，文本编码器的输出
        text_features = model.text_encoder(text_tokens)

    # 遍历测试集中的图片和标签
    for img, label in test_loader:
        img_input = img.to(device)  # 图片输入转到设备
        with torch.no_grad():
            # 计算图像特征，图像编码器的输出
            image_feature = model.image_encoder(img_input)

        # 模型里的logit_scale参数通常是对数，exp后得到缩放因子
        logit_scale = model.logit_scale.exp()
        # 计算图像特征和所有类别文本特征的相似度(logits)
        logits = logit_scale * image_feature @ text_features.t()
        # 通过softmax转成概率分布
        probs = logits.softmax(dim=-1)

        # 取概率最大的类别索引
        predicted_idx = probs.argmax(dim=-1).item()
        predicted_label = class_names[predicted_idx]

        # 输出真实类别和预测类别
        print(f"真实类别: {class_names[label]}    预测类别: {predicted_label}")
        break  # 这里只测试了第一个样本，实际可去掉break遍历所有

if __name__ == "__main__":
    evaluate()
