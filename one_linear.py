# 单特征值拟合

# 训练数据: x: “服务器数量”
#           y: “空调功率”
from helper.linear import *


def load_data():
    train_data = np.load("./data/one.npz")
    train_x = train_data["data"]
    label = train_data["label"]
    return train_x, label


if __name__ == "__main__":
    net = Network([1, 1], {"eta": 0.3, "batch_size": 10,
                           "max_epoch": 100, "eps": 0.02})
    net.train(load_data())
    # 预测 346 台服务器
    result = net.inference(0.346)
    print(f"result: {result}")
