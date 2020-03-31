# *_* coding:utf-8 *_*
# @author:sdh
# @Time : 2020/3/26 0026 16:50
import numpy as np

# 1 numpy中的数据类型
# 布尔类：bool、
# 整型：int8、int16、int32、int64、uint8、 uint16、 uint32、 uint64、
# 浮点型：float16、float32、float64、
# 等等。。。

# print("数据类型是：",np.dtype(np.int32))
# print("数据类型是：",np.dtype(np.float32))

# 2 numpy中数组的属性
# 2.1 数组的维度
a1 = np.arange(24)
# print(a1.ndim)  # 1
a2 = a1.reshape(2,3,4)
# print(a2.ndim)  # 3

# 2.2 数组的长宽高等尺寸
a3 = np.array([[1,2,3],[1,2,3]])
# print(a3.shape) # (2,3)
# print(a3.ndim) # 2
a4 = np.array([[1,2,3]])
# print(a4.shape) # (1,3)
# print(a4.ndim) # 2

a5 = np.array([
    [
        [1,2,3,4],[1,2,3,4],[1,2,3,4]
        ],
    [
        [1,2,3,4],[1,2,3,4],[1,2,3,4]
        ]
    ])
# print(a5.shape) # (2,3,4)
# print(a5.ndim) # 3
# print(a5.dtype) # int32/float64

# 2.3 数组中元素的字节数
a6 = np.array([1.0,2,3],dtype=np.float32)
# print(a6.itemsize)   # int32：4/float32：4/float64：8

# 2 numpy中如何创建数组
# 2.1 创建一个空的数组
b1 = np.empty(shape=[3,2],dtype=np.float32)
# print(b1)  # 随机值，未初始化

b2 = np.empty(shape=[3,2],dtype=np.int32)
# print(b2)  # 随机值，未初始化 0

b3 = np.empty((3,2),dtype=np.float32)
# print(b3)  # 随机值，未初始化

b4 = np.empty((3,2),dtype=np.int32)
# print(b4)  # 随机值，未初始化 0

# 2.2 创建全0数组
b5 = np.zeros((4,4),dtype=np.float32)
# print(b5)

b6 = np.zeros_like(b5)
# print(b6)

# 2.3 创建全1数组
b7 = np.ones((5,5),dtype=np.int32)
# print(b7)

b8 = np.ones_like(b7)
# print(b8)

# 2.4 将列表和元组转为数组
x = [1,2,3]
y= (1,2,3)
b9 = np.asarray(x)
b10 = np.asarray(y)
# print(b9)
# print(b10)

# 2.5 从数值范围创建数组
c1 = np.arange(5)
# print(c1) # [0 1 2 3 4]

c2 = np.arange(1,5)
# print(c2) # [1 2 3 4]

c3 = np.arange(1,5,step=2)
# print(c3) # [1 3]

c4 = np.linspace(1,10,num=10)  # 构造等差数列
# print(c4)

c5 = np.linspace(1,10,num=5)  # 构造等差数列：（stop-start）/(num-1)=2.25
# print(c5)

c6 = np.linspace(1,10,num=5,endpoint=False)  # 构造等差数列：（stop-start）/num=1.8
# print(c6)

c7 = np.logspace(1,2,num=10)   # 构造对数等比数列：底为10
c8 = np.logspace(1,2,num=10,base=2)   # 构造对数等比数列：底为2

# 3 数组切片与索引
# 3.1 数组切片

d= np.arange(10)

# print(d[1:5])  # [1,2,3,4]
# print(d[:5])  # [0,1,2,3,4]
# print(d[1:5:2])  # [1,3] 间隔1个
# print(d[1:5:3])  # [1,4] 间隔2个

# print(d[::-1])
# print(d[-1::-1])
# print(d[-1:0:-1])
# print(d[-1:0:-1][::2])

d1 = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])

# print(d1)  # 全部
# print(d1[:,:]) # 全部

# print(d1[0,:])   # 第一行的全部

# print(d1[1:])   # 除第一行的全部
# print(d1[1:,:]) # 除第一行的全部

# print(d1[1:,1:])  # 除第一行和第一列的全部

# print(d1[:,0])  # 第一列的全部
# print(d1[:,0:2])  # 第一列的某些

# print(d1[[0,0],[1,1]])
# print(d1[d1>3])

# 4 数组的广播
# g1 = np.arange(1,5)
# g2 = np.arange(10,50,step=10)

# print(g1*g2)
# g1 = np.array([
#     [ 0, 0, 0],
#     [10,10,10],
#     [20,20,20],
#     [30,30,30]
#     ])
# g2 = np.array([1,2,3])

# print(g1+g2)

# 5 数组的迭代
h = np.arange(6).reshape(2,3)
# for i in np.nditer(h.T):
#     print(i)


# 6 数组操作
# 6.1 修改数组形状reshape
# print(h.reshape(3,2))
# 6.2 数组展平 flatten
# print(h.flatten())

# 6.3 数组转置
# print(h.T)

# 6.4 数组扩展维度expand_dims

# 6.4 数组删除维度为1的条目(1,2,3)---->(2,3)

# 6.5 数组的连接
# np.concatenate((a,b),axis)
f1 = np.array([[1,2],[3,4]])
f2 = np.array([[5,6],[7,8]])
# print (np.concatenate((f1,f2)))
# print (np.concatenate((f1,f2),axis=1))

# np.stack((a,b),axis)
print (np.stack((f1,f2)))
print (np.stack((f1,f2),axis=1))

# np.hstack((a,b))
# np.vstack((a,b))

# 6.6 数组切分 split
# 6.7 数组中去除重复元素unique

# 7 数组中的数学运算
# 7.1  三角函数：sin()、cos()、tan()。
# 7.2  around() 函数返回指定数字的四舍五入值。
# 7.3  floor() 返回小于或者等于指定表达式的最大整数，即向下取整。
# 7.4  ceil() 返回大于或者等于指定表达式的最小整数，即向上取整。
# 7.5  加减乘除: add()，subtract()，multiply() 和 divide()。
# 7.6 mod() 计算输入数组中相应元素的相除后的余数。 函数 numpy.remainder() 也产生相同的结果。
# 7.7  reciprocal() 函数返回参数逐元素的倒数
# 7.8  reciprocal() 函数返回参数逐元素的倒数
# 7.9  power()函数将第一个输入数组中的元素作为底数，计算它与第二个输入数组中相应元素的幂。

# 8 数组中的统计函数
# 8.1  amin() 用于计算数组中的元素沿指定轴的最小值
# 8.2  amax() 用于计算数组中的元素沿指定轴的最大值
# 8.3  ptp()函数计算数组中元素最大值与最小值的差（最大值 - 最小值）
# 8.4  median() 函数用于计算数组 a 中元素的中位数（中值）
# 8.5  mean() 函数返回数组中元素的算术平均值
# 8.6  average() 函数根据在另一个数组中给出的各自的权重计算数组中元素的加权平均值
# 8.7  var() 统计中的方差（样本方差）是每个样本值与全体样本值的平均数之差的平方值的平均数
# 8.8  percentile()百分位数是统计中使用的度量，表示小于这个值的观察值的百分比

# 9 排序
# 9.1 sort()函数,排序方法：'quicksort'（快速排序）'mergesort'（归并排序） 'heapsort'（堆排序）
# 9.2 lexsort() 用于对多个序列进行排序。把它想象成对电子表格进行排序，每一列代表一个序列，排序时优先照顾靠后的列

# 10 线性代数
# 10.1 dot() 对于两个一维的数组，计算的是这两个数组对应下标元素的乘积和(数学上称之为内积)
# 10.2 matmul() 函数返回两个数组的矩阵乘积
# 10.3 numpy.linalg.det() 函数计算输入矩阵的行列式
# 10.4 numpy.linalg.solve() 函数给出了矩阵形式的线性方程的解
# 10.5 numpy.linalg.inv() 函数计算矩阵的乘法逆矩阵。

# 11 numpy IO
# 11.1 load()、save() 函数是读写文件数组数据的两个主要函数，默认情况下，数组是以未压缩的原始二进制格式保存在扩展名为 .npy 的文件中
# 11.2 savze() 函数用于将多个数组写入文件，默认情况下，数组是以未压缩的原始二进制格式保存在扩展名为 .npz 的文件中。
# 11.3 loadtxt() 和 savetxt() 函数处理正常的文本文件(.txt 等)
