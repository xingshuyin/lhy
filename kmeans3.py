import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class kmeans():
    def __init__(self,data=None, n_site=4, dim=2, max_iter=5000,tol=1e-4, distance_method=None,dtype=np.int8, select_choice=True) -> None:
        self.n_site= n_site
        self.dim=dim
        self.max_iter=max_iter
        self.tol=tol
        if data:
            if type(data)==list:
                
                self.data=np.array(data, dtype=dtype)
                if select_choice:
                    lines = np.random.choice(range(self.data.shape[0]), self.n_site, replace=False)
                    self.sites=self.data[lines] # 随机选择n_site个点为初始站点
                else:
                    self.init_sites()
            self.dim = len(data[0])
        self.distance_method=distance_method
    def init_sites(self):
        np.random.randint(0,5,(self.n_site,self.dim),dtype=np.float32)
    def distance(self, site, item):
        if self.distance_method:
            return self.distance_method(site, item)
        return  np.sqrt(np.sum(np.square(site - item)))
    def get_distances(self):
        r = np.array([np.sqrt(np.sum(np.square(self.data - i),axis=1)) for i in self.sites])
        # print(r)
        min_distances = r.min(axis=0)
        min_indexs = r.argmin(axis=0)
        self.groups = [[] for i in range(self.n_site)]
        for index,i in enumerate(self.data):
            self.groups[min_indexs[index]].append([index,min_distances[index]]) 
    
    def update_sites(self):
        for index,i in enumerate(self.groups):
            items = self.data[[j[0] for j in i]]
            # print(items)
            self.sites[index] = np.mean(items, axis=0)
            # print(np.mean(items, axis=0))
    
    def forward(self):
        for i in range(self.max_iter):
            self.get_distances()
            self.update_sites()
        print(self.groups)
        print(self.sites)
    
    def group_data(self):
        r = []
        for i in self.groups:
            r.append(self.data[[j[0] for j in i]])
        return r
if __name__=="__main__":
    data = [[1, 28, 2.608695652173913, [20, 23]], [20, -47, 2.869565217391304, [22, 25]], [-33, -41, 0.391304347826087, [9, 10]], [18, -8, 1.0434782608695652, [12, 14]], [-37, 10, 3.130434782608696, [24, 27]], [-48, -6, 0.782608695652174, [9, 11]], [-45, 16, 0.43478260869565216, [5, 7]], [6, 33, 0.0, [0, 1]], [7, 1, 0.30434782608695654, [7, 8]], [41, -28, 0.13043478260869565, [1, 4]], [29, 7, 1.1304347826086956, [13, 15]], [42, -42, 0.34782608695652173, [4, 6]], [12, -46, 1.3043478260869565, [10, 13]], [-25, -7, 1.826086956521739, [21, 23]], [-35, -41, 2.608695652173913, [20, 23]], [28, 38, 0.5217391304347826, [4, 7]], [43, -17, 2.217391304347826, [17, 20]], [-43, -2, 0.043478260869565216, [1, 2]], [-31, -1, 0.782608695652174, [9, 11]], [3, -8, 1.0434782608695652, [12, 14]]]
    k = kmeans(data=[i[:3] for i in data],n_site=4, dim=3,dtype=np.float64)
    k.forward()
    r = k.group_data()
    print(r)
    colors = [
'black',
'blue',
'yellow',
'pink',
    ]
    p = plt.subplot(projection = '3d')
    for index, g in enumerate(r):
        x = [i[0] for i in g]
        y = [i[1] for i in g]
        z = [i[2] for i in g]
        
        print(colors[index])
        p.scatter(x,y,z, c=colors[index], s=10)
    x = [i[0] for i in k.sites]
    y = [i[1] for i in k.sites]
    z = [i[2] for i in k.sites]
    p.scatter(x,y,z, c='black', s=40, marker=10)
    p.set(xlabel='x', ylabel='y',zlabel='z')
    # p.viewLim()
    plt.show()