# *_* coding:utf-8 *_*
# @author:sdh
# @Time : 2020/3/31 0031 14:27
"""
hidden_layer_sizes : 默认(100，），第i个元素表示第i个隐藏层的神经元的个数,如(100,100)表示两层，每层100个神经元。

activation :{‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, 默认‘relu
- ‘identity’： no-op activation, useful to implement linear bottleneck，
返回f(x) = x
- ‘logistic’：the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
- ‘tanh’：the hyperbolic tan function, returns f(x) = tanh(x).
- ‘relu’：the rectified linear unit function, returns f(x) = max(0, x)

solver： {‘lbfgs’, ‘sgd’, ‘adam’}, 默认 ‘adam’，用来优化权重
- lbfgs：quasi-Newton方法的优化器
- sgd：随机梯度下降
- adam： Kingma, Diederik, and Jimmy Ba提出的机遇随机梯度的优化器
注意：默认solver ‘adam’在相对较大的数据集上效果比较好（几千个样本或者更多），对小数据集来说，lbfgs收敛更快效果也更好。

alpha :float,可选的，默认0.0001,正则化项参数

batch_size : int , 可选的，默认‘auto’,随机优化的minibatches的大小，如果solver是‘lbfgs’，分类器将不使用minibatch，当设置成‘auto’，batch_size=min(200,n_samples)

learning_rate :{‘constant’，‘invscaling’, ‘adaptive’},默认‘constant’，用于权重更新，只有当solver为’sgd‘时使用
- ‘constant’: 有‘learning_rate_init’给定的恒定学习率
- ‘incscaling’：随着时间t使用’power_t’的逆标度指数不断降低学习率learning_rate_ ，effective_learning_rate = learning_rate_init / pow(t, power_t)
- ‘adaptive’：只要训练损耗在下降，就保持学习率为’learning_rate_init’不变，当连续两次不能降低训练损耗或验证分数停止升高至少tol时，将当前学习率除以5.

max_iter: int，可选，默认200，最大迭代次数。

random_state:int 或RandomState，可选，默认None，随机数生成器的状态或种子。

shuffle: bool，可选，默认True,只有当solver=’sgd’或者‘adam’时使用，判断是否在每次迭代时对样本进行清洗。

tol：float, 可选，默认1e-4，优化的容忍度

learning_rate_int:double,可选，默认0.001，初始学习率，控制更新权重的补偿，只有当solver=’sgd’ 或’adam’时使用。

power_t: double, optional, default 0.5，只有solver=’sgd’时使用，是逆扩展学习率的指数.当learning_rate=’invscaling’，用来更新有效学习率。

verbose : bool, optional, default False,是否将过程打印到stdout

warm_start : bool, optional, default False,当设置成True，使用之前的解决方法作为初始拟合，否则释放之前的解决方法。

momentum : float, default 0.9,Momentum(动量） for gradient descent update. Should be between 0 and 1. Only used when solver=’sgd’.

nesterovs_momentum : boolean, default True, Whether to use Nesterov’s momentum. Only used when solver=’sgd’ and momentum > 0.

early_stopping : bool, default False,Only effective when solver=’sgd’ or ‘adam’,判断当验证效果不再改善的时候是否终止训练，当为True时，自动选出10%的训练数据用于验证并在两步连续爹迭代改善低于tol时终止训练。

validation_fraction : float, optional, default 0.1,用作早期停止验证的预留训练数据集的比例，早0-1之间，只当early_stopping=True有用

beta_1 : float, optional, default 0.9，Only used when solver=’adam’，估计一阶矩向量的指数衰减速率，[0,1)之间

beta_2 : float, optional, default 0.999,Only used when solver=’adam’估计二阶矩向量的指数衰减速率[0,1)之间

epsilon : float, optional, default 1e-8,Only used when solver=’adam’数值稳定值。
属性说明：
- classes_:每个输出的类标签
- loss_:损失函数计算出来的当前损失值
- coefs_:列表中的第i个元素表示i层的权重矩阵
- intercepts_:列表中第i个元素代表i+1层的偏差向量
- n_iter_ ：迭代次数
- n_layers_:层数
- n_outputs_:输出的个数
- out_activation_:输出激活函数的名称。

"""
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

digits = load_digits()
x = digits.data
y = digits.target
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3)
cls = MLPClassifier(activation='relu',
                    alpha=1e-05,
                    batch_size='auto',
                    beta_1=0.9,
                    beta_2=0.999,
                    early_stopping=False,
                    epsilon=1e-08,
                    hidden_layer_sizes=(50, 50, 50),
                    learning_rate='constant',
                    learning_rate_init=0.001,
                    max_iter=500,
                    momentum=0.9,
                    nesterovs_momentum=True,
                    power_t=0.5,
                    random_state=1,
                    shuffle=True,
                    solver='lbfgs',
                    tol=0.0001,
                    validation_fraction=0.1,
                    verbose=False,
                    warm_start=False)
cls.fit(x_train,y_train)
s = cls.score(x_test,y_test)
print(s)
print('准确率： %s' % cross_val_score(cls, x, y, cv=5).mean())
