from math import *  
from random import *
import numpy as np
import matplotlib.pyplot as plt
 # https://blog.csdn.net/m0_74061077/article/details/129882695?spm=1001.2014.3001.5502
#遗传算法步骤
f=lambda x:x+10*np.sin(5*x)+7*np.cos(4*x)
 
#绘制函数图像,读者可自行去掉#号查看
x0=np.linspace(0,10,50000)
y0=f(x0)
plt.figure()
plt.plot(x0,y0)
plt.show()
 
#初始参数
num=100 #种群数量
n=20 #DNA长度
pc=0.6 #交叉互换概率
pm=0.001 #变异概率
G=100 #最大代数
adaption=0.01 #适应度补充值
x=[i for i in range(0,101)] #代数，即迭代次数
y=[] #历代最优值
a=0 #下界
b=10 #上界
lis=[] #种群集
 
#1.选取初始种群， 生成num个种群
for i in range(num):
    lis.append(np.random.randint(2,size=n))  # 生成n个0-2的随机数不包括2  的种群
    #产生num个随机的DNA，只用两种碱基
 
for generation in range(G): # 迭代G代
    #2.解码
    lis0=[] #个体函数值集，保存所有个体的十进制值
    for k in lis: # 遍历种群集
        s=0
        for j in range(n): #遍历一个种群,  假设上面lis里的种群里是生成的各位数二进制值， 那这句话就是就按二进制值的十进制值
            s+=k[j]*2**(n-1-j)
        lis0.append(f(a+s*(b-a)/(2**n)))
    
    #3.适应度计算
    minlis=min(lis0)
    maxlis=max(lis0)
    y.append(maxlis) # 把最大值添加到最大值列表中
    top=lis0.index(maxlis)  # 获取最大值的索引
    ad_lis=[] #适应度集
    for i in range(num): # 保证适应度大于0
        ad_lis.append(lis0[i]-minlis+adaption) # 保证适应度大于0
    sum_ad=sum(ad_lis)
    for i in range(num): # 适应度归一化， 可用于比较权重或概率
        ad_lis[i]=ad_lis[i]/sum_ad
        
    #4.物竞天择
    #先计算部分和方便使用轮盘法
    ad_k=[] #适应度部分和序列， 前I个种群的和
    optlis=[] #储存物竞天择后的个体
    for i in range(num):
        ad_k.append(sum(ad_lis[0:i+1]))
    for i in range(num-1):#物竞天择,但是留出一个位置给父辈的最优者
        p=random() # 生成一个【0，1）的值
        for j in range(num): # 随机选一个值，类似轮盘读选择法
            if p<ad_k[j]:
                opt=j
                break
        optlis.append(lis[opt].copy()) # 把随机选择的个体（种群）加到物竞天择的列表中
        #lis[opt]是对lis中的数组的引用！！
        #这里如果不使用copy,后面对optlis进行修改时也会带动lis的修改！！！
 
    #5.繁衍后代
    #先交叉互换，然后变异
    newlis=[]
    for i in range(0,num-1): # 根据交叉互换概率，随机交换两个样本的某个位置的基因
        A=optlis[i]#父本
        l=randint(0,98)
        B=optlis[l]#母本
        for j in range(0,n):
            p1=random()
            if p1<pc: # 大于随机值就交换， 
                A[j]=B[j] #交叉互换
        newlis.append(A.copy())
    
    for i in range(0,num-1):
        C=newlis[i]
        for j in range(0,n):
            p2=random()
            if p2<pm:
                C[j]=1-C[j] #变异
    newlis.append(lis[top].copy())
 
    #6.重复进化
    lis=newlis
    #将上面的步骤重复G次，即繁衍G代
 
lis0=[] #清空上一代函数值，准备做最后的处理
x1=[] #最后一代x值集合   
for k in lis:
    s=0
    for i in range(n):
        s+=k[i]*2**(n-1-i)
    x1.append(a+s*(b-a)/(2**n)) #记录最后一代的x值
    lis0.append(f(a+s*(b-a)/(2**n))) #记录最后一代的y值
y.append(max(lis0)) #将最后一代的最优值储存
top=lis0.index(max(lis0))
print([x1[top],max(lis0)]) #输出最优x值和y值
 
#绘制图像
yc=[max(y0) for i in range(len(y))]
plt.figure()
plt.plot(x,y,'b-',label='GA')
plt.plot(x,yc,'r-',label='max(f)')
plt.legend()
plt.show()