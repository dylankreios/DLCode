# 二特征值拟合
# 与单特征值不同，要考虑标准化

# 训练数据: x: “距市中心距离”以及“房屋面积”
#           y: “北京通州房价”
from helper.normalize import *
from helper.linear import *


def load_data():
    # 加载数据
    train_data = np.load("./data/multi.npz")
    train_x = train_data["data"]
    label = train_data["label"]
    return train_x, label


if __name__ == "__main__":
    net = Network([2, 1], {"eta": 0.01, "batch_size": 10,
                           "max_epoch": 200, "eps": 1e-5})
    data = load_data()
    x_new = normalize_x(data)[0]
    y_new = normalize_y(data)[0]
    net.train((x_new, y_new))

    X_norm = normalize_x(data)[1]
    Y_norm = normalize_y(data)[1]

    x = np.array([15, 93]).reshape(1, 2)
    x_pred_new = normalize_x_pred(x, X_norm)
    z = net.inference(x_pred_new)
    # 还原预测输出
    Z_true = z * Y_norm[0, 1] + Y_norm[0, 0]
    print(f"------------------\nz: {z}\nZ_true: {Z_true}")
