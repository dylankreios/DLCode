# 二分类（训练数据已标准化）

# 训练数据: x: “坐标点（0.0 - 1.1）”
#           y: “汉 or 楚”
from helper.classify import *
from helper.classify_tanh import *


def load_data():
    train_data = np.load("data/binary.npz")
    train_x = train_data["data"]
    label = train_data["label"]
    return train_x, label


def load_data_tanh():
    train_data = np.load("data/binary.npz")
    train_x = train_data["data"]
    label = train_data["label"]

    new_label = np.zeros(label.shape)
    for i in range(label.shape[0]):
        if label[i, 0] == 0:     # 第一类的标签设为0
            new_label[i, 0] = -1
        elif label[i, 0] == 1:   # 第二类的标签设为1
            new_label[i, 0] = 1

    return train_x, new_label


if __name__ == "__main__":
    # net = Network([2, 1], {"eta": 0.1, "batch_size": 10,
    #                        "max_epoch": 10000, "eps": 1e-3, "net_type": NetType.BinaryClassifier})
    net = TanhNetwork([2, 1], {"eta": 0.1, "batch_size": 10,
                               "max_epoch": 1000, "eps": 1e-3, "net_type": NetType.BinaryTanh})
    data = load_data_tanh()
    net.train(data, check_point=10)
