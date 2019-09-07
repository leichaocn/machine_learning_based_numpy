import numpy as np

from .dt import DecisionTree
from .losses import MSELoss, CrossEntropyLoss

"""定义了一个梯度提升决策树



"""


def to_one_hot(labels, n_classes=None):
    if labels.ndim > 1:
        raise ValueError("labels must have dimension 1, but got {}".format(labels.ndim))
    """
    one_hot的尺寸为（N,n_cols）
    行数等于Y（或X）的样本总行数，即N。
    列数等于Y的类别数，或者是最大类别数字加1（因为默认所有类别的编号是从0递增到n_classes-1）


    """
    N = labels.size
    n_cols = np.max(labels) + 1 if n_classes is None else n_classes
    one_hot = np.zeros((N, n_cols))
    """
    假如有N=5个样本，
    np.arange(N)=np.array([0,1,2,3,4])，
    labels=np.array([3,2,1,3,1])
    依次组合这两个数组的对应值，来给one_hot的零矩阵对应行列赋值为1。
    one_hot= [[0. 0. 0. 1.]           （0,3）
            [0. 0. 1. 0.]            （1,2）
            [0. 1. 0. 0.]              （2,1）
            [0. 0. 0. 1.]                （3,3）
            [0. 1. 0. 0.]]                 （4,1）
    最终形成，对于每个样本，即每一行，1所在的索引位置，代表本条样本对应的label编号
    """
    one_hot[np.arange(N), labels] = 1.0
    return one_hot


class GradientBoostedDecisionTree:
    """
    An instance of gradient boosted machines (GBM) using decision trees as the
    weak learners.

    GBMs fit an ensemble of m weak learners s.t.:

        f_m(X) = b(X) + lr * w_1 * g_1 + ... + lr * w_m * g_m

    where b is a fixed initial estimate for the targets, lr is a learning rate
    parameter, and w* and g* denote the weights and learner predictions for
    subsequent fits.

    We fit each w and g iteratively using a greedy strategy so that at each
    iteration i,

        w_i, g_i = argmin_{w_i, g_i} L(Y, f_{i-1}(X) + w_i * g_i)

    On each iteration we fit a new weak learner to predict the negative
    gradient of the loss with respect to the previous prediction, f_{i-1}(X).
    We then use the element-wise product of the predictions of this weak
    learner, g_i, with a weight, w_i, to compute the amount to adjust the
    predictions of our model at the previous iteration, f_{i-1}(X):

        f_i(X) := f_{i-1}(X) + w_i * g_i
    """

    def __init__(
        self,
        n_iter,
        max_depth=None,
        classifier=True,
        learning_rate=1,
        loss="crossentropy",
        step_size="constant",
        split_criterion="entropy",
    ):
        """
        A gradient boosted ensemble of decision trees.

        Parameters
        ----------
        n_iter : int
            The number of iterations / weak estimators to use when fitting each
            dimension/class of Y
        max_depth : int (default: None)
            The maximum depth of each decision tree weak estimator
        classifier : bool (default: True)
            Whether Y contains class labels or real-valued targets
        learning_rate : float (default: 1)
            Value in [0, 1] controlling the amount each weak estimator
            contributes to the overall model prediction. Sometimes known as the
            `shrinkage parameter` in the GBM literature
        loss : str
            The loss to optimize for the GBM. Valid entries are {"crossentroy",
            "mse"}.
        step_size : str
            How to choose the weight for each weak learner. Valid entries are
            {"constant", "adaptive"}. If "constant", use a fixed weight of 1
            for each learner. If "adaptive", use a step size computed via
            line-search on the current iteration's loss.
        split_criterion : str
            The error criterion to use when calculating splits for each weak
            learner. When `classifier` is False, valid entries are {'mse'}.
            When `classifier` is True, valid entries are {'entropy', 'gini'}.
        """
        # 交叉熵 or MSE ？
        self.loss = loss
        self.weights = None
        self.learners = None
        self.out_dims = None
        self.n_iter = n_iter
        self.base_estimator = None
        # 单一决策树的最大深度限制
        self.max_depth = max_depth
        self.step_size = step_size
        # 分类 or 回归
        self.classifier = classifier
        self.learning_rate = learning_rate
        # 单一决策树训练时使用的分割指标，mse or entropy gini
        self.split_criterion = split_criterion

    def fit(self, X, Y):
        """
        1.选loss
        2.改造Y
        3.初始化学习器矩阵、权重矩阵
        4.训练首批学习器，即填充学习器矩阵的第一行
        5.
        """
        if self.loss == "mse":
            loss = MSELoss()
        elif self.loss == "crossentropy":
            loss = CrossEntropyLoss()

        # convert Y to one_hot if not already
        if self.classifier:
            """ Y   =     [[0. 0. 0. 1.]
                        [0. 0. 1. 0.]
                        [0. 1. 0. 0.]
                        [0. 0. 0. 1.]  
                        [0. 1. 0. 0.]]
                shape=(5, 4)   
            """
            Y = to_one_hot(Y.flatten())
        else:
            """对于回归问题，将Y规范成尺寸(N,1)的数组.
            如果Y本身尺寸正是(N,1)，此时len((N,1))=2，此时不做任何处理。
            如果Y本身尺寸正是(N,)，此时len((N,))=1，此时reshape列为1列，将被转为尺寸(N,1)
            假设
            Y=np.array([3,2,1,3,1])，shape=(5,)
            reshape后
            Y= [[3]
                 [2]
                 [1]
                 [3]
                 [1]]
            shape=(5, 1)
            """
            Y = Y.reshape(-1, 1) if len(Y.shape) == 1 else Y

        # N为样本总数，M为特征总数
        N, M = X.shape
        # out_dims 存储Y的列数
        self.out_dims = Y.shape[1]

        """学习器数组,后面用于保存若干个base_estimator对象
        每一行表示一个base_estimator，行数为弱分类器的个数 n_iter
        每一列表示base_estimator输出的一个维度，列数为Y的列数 out_dims
        """
        self.learners = np.empty((self.n_iter, self.out_dims), dtype=object)

        # 权重数组，尺寸同上，初始化为1
        self.weights = np.ones((self.n_iter, self.out_dims))

        """除了第一个学习器，后面的学习器的权重都乘以学习率
        例如一个3个学习器，Y输出维度为4的权重数组。
        weights = np.ones((3, 4))
        weights[1:, :] *= 0.4
        结果为：
        weights =  [[1.  1.  1.  1. ]   
                    [0.4 0.4 0.4 0.4]
                    [0.4 0.4 0.4 0.4]]
        """
        self.weights[1:, :] *= self.learning_rate

        # fit the base estimator
        """Y_pred 是一个尺寸为（N,out_dims）,即每一行对应一个样本，每一列对应一个Y的预测输出
        """
        Y_pred = np.zeros((N, self.out_dims))
        for k in range(self.out_dims):
            """针对Y的每一个列的label，训练一个base_estimator
            训练数据：特征为X，target为Y数据的某一列。
            总共训练out_dims个，并把预测结果存入Y_pred的对应列
            最终获得的是：
            Y_pred 每一列是一个estimater对全量样本X的预测序列
            learners[0, k]为第0次
            """
            t = loss.base_estimator()
            t.fit(X, Y[:, k])
            # 填充Y_pred的第k列。这里的+毫无意义，因为本身就是0。
            Y_pred[:, k] += t.predict(X)
            # learners的每一行，保存着out_dims个base_estimator对象。
            self.learners[0, k] = t

        # incrementally fit each learner on the negative gradient of the loss
        # wrt the previous fit (pseudo-residuals)
        for i in range(1, self.n_iter):
            for k in range(self.out_dims):
                """ 
                假设y、y_pred,
                y = np.array([0.1,1])
                y_pred = np.array([2,0.9])
                
                算出来的负梯度为
                neg_grad = [-0.05       -1.11111111]
                那么训练下一个基估计器，让它预测为 -0.05  -1.111
                
                这样f=[2   0.9]+[-0.05       -1.111]
                """
                y, y_pred = Y[:, k], Y_pred[:, k]
                neg_grad = -1 * loss.grad(y, y_pred)

                # use MSE as the surrogate loss when fitting to negative gradients
                """训练回归树
                """
                t = DecisionTree(
                    classifier=False, max_depth=self.max_depth, criterion="mse"
                )

                # fit current learner to negative gradients
                t.fit(X, neg_grad)
                self.learners[i, k] = t

                # compute step size and weight for the current learner
                step = 1.0
                h_pred = t.predict(X)
                if self.step_size == "adaptive":
                    step = loss.line_search(y, y_pred, h_pred)

                # update weights and our overall prediction for Y
                self.weights[i, k] *= step
                Y_pred[:, k] += self.weights[i, k] * h_pred

    def predict(self, X):
        """构造一个全零数组Y_pred，行数等于X里的样本总数，列数等于target的输出维度out_dims。
        假设有2个样本，out_dims是4维，则
        Y_pred =  [[0. 0. 0. 0.]
                   [0. 0. 0. 0.]]

        假设学习率为 0.7，
        基学习器数量 n_iter=3
        输出维度 out_dims=4，
        则
        weights =  [[1.  1.  1.  1. ]
                    [0.7 0.7 0.7 0.7]
                    [0.7 0.7 0.7 0.7]]
        learners = [[e00 e01 e02 e03]
                    [t10 t11 t12 t13]
                    [t20 t21 t22 t23]]
        """
        Y_pred = np.zeros((X.shape[0], self.out_dims))
        for i in range(self.n_iter):
            """
            当i=0，第一个估计器，
            因为X有两个样本，则self.learners[i, k].predict(X)返回的是[v1, v2],或者表示为e00序列
            循环out_dims=4里的所有k，结果为
            Y_pred=[[1*e00[0]  1*e01[0]   1*e02[0]     1*e03[0]]
                    [1*e00[1]  1*e01[1]    1*e02[1]    1*e03[1]]
                    ]

            当i=1，第二个估计器，
            因为X有两个样本，则self.learners[i, k].predict(X)返回的是[v1, v2],或者表示为e00序列
            循环out_dims=4里的所有k，结果为
            Y_pred=[[1*e00[0]+0.7*t10[0]   1*e01[0]+0.7*t11[0]      1*e02[0]+0.7*t12[0]        1*e03[0]+0.7*t13[0]   ]
                    [1*e00[1]+0.7*t10[1]   1*e01[1]+0.7*t11[1]       1*e02[1]+0.7*t12[1]       1*e03[1]+0.7*t13[1]   ]
                    ]

            当i=2，第三个估计器，
            因为X有两个样本，则self.learners[i, k].predict(X)返回的是[v1, v2],或者表示为e00序列
            循环out_dims=4里的所有k，结果为
            Y_pred=[[1*e00[0]+0.7*t10[0]+0.7*t20[0]   1*e01[0]+0.7*t11[0]+0.7*t21[0]      1*e02[0]+0.7*t12[0]+0.7*t22[0]       1*e03[0]+0.7*t13[0]+0.7*t23[0]     ]
                    [1*e00[1]+0.7*t10[1]+0.7*t20[1]   1*e01[1]+0.7*t11[1]+0.7*t21[1]      1*e02[1]+0.7*t12[1]+0.7*t22[1]       1*e03[1]+0.7*t13[1]+0.7*t23[1]     ]
                    ]
            """
            for k in range(self.out_dims):
                Y_pred[:, k] += self.weights[i, k] * self.learners[i, k].predict(X)

        """总结：关于 Y_pred
                1.行数为样本的数目，列数为target的维度。
                2.有多少个估计器 n_iter ，Y_pred每一个元素就是多少个迭代器的叠加。
                
                关于叠加：
                1.权重矩阵和学习器矩阵同样的尺寸，即行数为估计器的生成批次，列数为输出的维度。
                2.权重矩阵的元素为标量，第一行均为1，其他行均为学习率；
                3.学习器矩阵的元素均为模型对象，第一行是基学习器，其他行是DT。每一行的学习器是同一批次但完全相同的模型。
                给单个模型对象填入X，输出行数即为X的行数，即样本个数，而列数始终为1.
                4.如果只是为了一个“基础款”的Y_pred，那只需要权重矩阵的第一行与学习器矩阵第一行相乘，然后输出即可。
                  因为学习器的输出行数与样本个数相同，这保证了输出的Y_pred的行数为样本个数。
                5.因为我们是为了更精确的输出，所以才会给每一个元素继续叠加下一行带权重模型的输出。
                  叠加次数就是学习器的总训练批次数，即权重矩阵和学习器矩阵的行数。
        """

        """如果是分类问题，返回一个一维数组，元素代表每个样本对应的类别。
        由于之前对Y的one-hot处理，Y_pred也理应对每一条样本（即每一行），选择最大的那个值
        假如 Y_pred= [[1 2 3 4]
                    [3 2 1 0]]
        经过argmax(axis=1)后，
            Y_pred= [3 0] 
            即第一个样本的最大概率在第4列，第二个样本的最大概率出现在第1列。
        """
        if self.classifier:
            Y_pred = Y_pred.argmax(axis=1)

        return Y_pred
