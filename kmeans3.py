import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans

count_people = 80
count_car = 5
count_plane = 6
size_car = 200  # 车装载量
size_plane = 8  # 无人机装载量
size_site = 200
size_sites = 3200  # 所有站点服务能力总和
size_same_pick = 8  # 在同一时段内配送站点的最大取货人数

speed_car = 40  # 汽车速度
speed_plane = 50  # 无人机速度

costs_trans_car = 1.5  # 车辆单位运输成本
costs_trans_plane = 0.5  # 无人机单位运输成本
costs_sites = 10  # 配送点固定成本
fresh_price = 25  # 生鲜单价
fresh_lose = 0.8  # 生鲜常温单位价值u说你好


just_num = 5 # 二进制位数

class kmeans():
    def __init__(self, data=None, n_site=4, dim=2, max_iter=5000, tol=1e-4, distance_method=None, dtype=np.int8, select_choice=True) -> None:
        self.n_site = n_site
        self.dim = dim
        self.max_iter = max_iter
        self.tol = tol
        if data:
            if type(data) == list:

                self.data = np.array(data, dtype=dtype)
                if select_choice:
                    lines = np.random.choice(range(self.data.shape[0]), self.n_site, replace=False)
                    self.sites = self.data[lines]  # 随机选择n_site个点为初始站点
                else:
                    self.init_sites()
            self.dim = len(data[0])
        self.distance_method = distance_method

    def init_sites(self):
        np.random.randint(0, 5, (self.n_site, self.dim), dtype=np.float32)

    def distance(self):
        if self.distance_method:
            return self.distance_method(self)
        return np.array([np.sqrt(np.sum(np.square(self.data - i), axis=1)) for i in self.sites])

    def get_groups(self):
        r = self.distance()
        # print(r)
        self.min_distances = r.min(axis=0)
        self.min_indexs = r.argmin(axis=0)
        self.groups = [[] for i in range(self.n_site)]
        for index, i in enumerate(self.data):
            self.groups[self.min_indexs[index]].append([index, self.min_distances[index]])

    def update_sites(self):
        for index, i in enumerate(self.groups):
            items = self.data[[j[0] for j in i]]
            # print(items)
            self.sites[index] = np.mean(items, axis=0)
            # print(np.mean(items, axis=0))

    def forward(self):  # 执行运算
        for i in range(self.max_iter):
            self.get_groups()
            self.update_sites()

    def group_data(self):  # 获取分组后的数据
        r = []
        for i in self.groups:
            r.append(self.data[[j[0] for j in i]])
        return r

    def SC(self):  # 获取所有点轮廓值
        # group_data = self.group_data()
        a = metrics.silhouette_samples(self.data, self.min_indexs)
        return a

    def show(self):
        r = self.group_data()
        colors = []
        p = plt.subplot(projection='3d')
        for index, g in enumerate(r):
            x = [i[0] for i in g]
            y = [i[1] for i in g]
            z = [i[2] for i in g]
            c = np.random.rand(1, 3)
            colors.append(c)
            p.scatter(x, y, z, c=c, s=5, marker='o')
        for index, i in enumerate(self.sites):
            p.scatter([i[0]], [i[1]], [i[2]], c=colors[index], s=40, marker=10)
        p.set(xlabel='x', ylabel='y', zlabel='z')
        # p.viewLim()
        plt.show()


def init(num):
    # points = []
    # for i in range(5):
    #     points.append([random.randint(-50,50),random.randint(-50,50)])
    # 生成居民
    peoples = []
    early_a = 24
    later_a = 0

    for i in range(num):
        early = random.randint(0, 25)  # 最早服务时间
        if early < early_a:
            early_a = early
        long = random.randint(1, 3)  # 服务间隔时长
        later = early + long
        if later <= 24:
            if later > later_a:
                later_a = later
        else:
            if later - 24 > later_a:
                later_a = later - 24

        peoples.append([random.randint(-50, 50), random.randint(-50, 50), [early, later], random.random() > 0.7, random.randint(2, 7)])
    for i in peoples:
        i.insert(2, i[2][0] * (i[2][1] - i[2][0]) / (later_a - early_a))  # 计算时间坐标
    return peoples  # [[x,y,z,[early,later], is_infect, value],]


def init_chain(peoples, groups=[], sites=[]):
    infect_peoples = [i for index, i in enumerate(peoples) if i[4]]
    # print(len(infect_peoples), infect_peoples)
    angles_infect = []
    angles_sites = []
    # p = [*infect_peoples, *sites]
    for index, i in enumerate(infect_peoples):
        # https://blog.csdn.net/chelen_jak/article/details/80518973
        a = math.atan2(i[1], i[0])
        angles_infect.append([a, index])
    angles_infect.sort(key=lambda x: x[0])
    for index, i in enumerate(sites):
        # https://blog.csdn.net/chelen_jak/article/details/80518973
        a = math.atan2(i[1], i[0])
        angles_sites.append([a, index])
    angles_sites.sort(key=lambda x: x[0])
    infect_order = [i[1] for i in angles_infect]
    sites_order = [i[1] for i in angles_sites]
    print('init_chain',infect_order, sites_order)
    chain_sites, chain_infect = make_chain(sites_order, infect_order)


def to(num, to_):
    r = []
    if num==0:
        return ['0']
    while True:
        num, y = divmod(num, to_)
        
        if num < to_:
            r.append(str(num))
            break
        else:
            r.append(str(y))
    return list(reversed(r))


def make_chain(sites_order, infect_order):
    sites_order_to = [''.join(to(i, 2)).rjust(just_num,'0') for i in sites_order]
    infect_orderr_to = [''.join(to(i, 2)).rjust(just_num,'0') for i in infect_order]
    print(sites_order_to, infect_orderr_to)
    sites_order_group = []
    indexs = random.sample(
        range(1,
              len(sites_order) - 1),
        count_car - 1,
    )
    index = 0
    # print(indexs)
    for i in sorted(indexs):
        sites_order_group.append(sites_order[index:i])
        index = i
    sites_order_group.append(sites_order[index:])

    infect_order_group = []
    indexs = random.sample(
        range(1,
              len(infect_order) - 1),
        count_plane - 1,
    )
    index = 0
    # print(indexs)
    for i in sorted(indexs):
        infect_order_group.append(infect_order[index:i])
        index = i
    infect_order_group.append(infect_order[index:])
    return sites_order_group, infect_order_group


def adapt(chain_sites, chain_infect):
    ...


if __name__ == "__main__":
    # data = [[1, 28, 2.608695652173913, [20, 23]], [20, -47, 2.869565217391304, [22, 25]], [-33, -41, 0.391304347826087, [9, 10]], [18, -8, 1.0434782608695652, [12, 14]], [-37, 10, 3.130434782608696, [24, 27]], [-48, -6, 0.782608695652174, [9, 11]], [-45, 16, 0.43478260869565216, [5, 7]], [6, 33, 0.0, [0, 1]], [7, 1, 0.30434782608695654, [7, 8]], [41, -28, 0.13043478260869565, [1, 4]], [29, 7,
    #  1.1304347826086956, [13, 15]], [42, -42, 0.34782608695652173, [4, 6]], [12, -46, 1.3043478260869565, [10, 13]], [-25, -7, 1.826086956521739, [21, 23]], [-35, -41, 2.608695652173913, [20, 23]], [28, 38, 0.5217391304347826, [4, 7]], [43, -17, 2.217391304347826, [17, 20]], [-43, -2, 0.043478260869565216, [1, 2]], [-31, -1, 0.782608695652174, [9, 11]], [3, -8, 1.0434782608695652, [12, 14]]]

    peoples = init(count_people)
    k = kmeans(data=[i[:3] for i in peoples], n_site=10, dim=3, max_iter=15000, dtype=np.float64)
    k.forward()
    min_indexs = k.min_indexs  # 每个点最短距离site(站点)
    groups = k.group_data()  # 分组结果
    sites = k.sites  # 站点坐标
    # k.show()
    # k.SC()
    init_chain(peoples, sites=sites, groups=groups)
