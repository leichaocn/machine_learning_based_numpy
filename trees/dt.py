import numpy as np


"""这段代码实现了决策树
定义了三个类：
一、Node：存储分割特征、分割点值，左右子树的对象
二、Leaf：训练时存储叶子节点的值，预测时返回该值。
三、DecisionTree：
    7个训练方法：
    fit(X,Y)：设置类别数 n_classes 和每次分割时的备选特征数n_feats，然后调用_grow(X, Y)获得一个树的根节点。
    _grow(X, Y)：递归式生长数，拿到树的跟节点。
                  1.每次随机拿几个特征feat_idxs，
                  2.调用_segment(X, Y, feat_idxs)获得最佳的分割特征和最佳的分割点值
                  3.递归式生长左右子树，
                  4.直到达到最大深度或者单一target序列，返回叶子节点。
                  5.最终返回一个根节点，保存有左右子树，分割点、分割点值。
    _segment(X, Y, feat_idxs)：遍历候选的几个特征，找出最佳特征及最佳分割点。
                    1.对于每个候选特征序列vals的每个候选分割点t，
                    调用_impurity_gain(Y, t, vals)，在计算信息增益。
                    2.返回具备最大信息增益的最佳特征及最佳分割点
    _impurity_gain(Y, t, vals)：获取信息增益值。
                    传入Y序列，某个候选特征序列，对应的某一分割点t
                    1.计算父节点的信息量
                    2.计算分割后的信息量
                    3.计算信息增益
                    信息增益越大，说明分割得越好。
    mse()：回归训练时的误差，越大说明该子树里真实数据越偏离预测值（均值），说明越凌乱，预测越拿不准。
    entropy()：分类器训练时的信息量，越大，代表信息量越大。说明越凌乱，预测越拿不准。
    gini():分类器训练时的信息量，越大，代表信息量越大。说明越凌乱，预测越拿不准。
    
    3个预测方法：
    predict():对传入的数据X，遍历每一条样本x，调用_traverse()
    predict_class_probs():同上，只是要求返回叶子的概率数组。
    _traverse(X, node):递归式判断传入的特征，在该节点的左or右子树，知道找到叶子节点，并返回。
"""

# 节点类
class Node:
    def __init__(self, left, right, rule):
        self.left = left
        self.right = right
        #  分割特征的id
        self.feature = rule[0]
        #  分割特征对应的分割点值
        self.threshold = rule[1]

# 叶子类
class Leaf:
    def __init__(self, value):
        """
        `value` is an array of class probabilities if classifier is True, else
        the mean of the region
        """
        # 此处的value，
        # 如果是分类器，是一个概率数组，即每个label对应的概率；返回时可以直接返回，也可以返回最大概率对应的索引值。
        # 如果是回归器，则是一个标量数值，即计算出的平均值。
        self.value = value


class DecisionTree:
    def __init__(
        self,
        # 回归 or 分类
        classifier=True,
        # 最大深度，如果填int值，则限制深度
        max_depth=None,
        # 设置特征数？
        n_feats=None,
        # 指标
        criterion="entropy",
        # 随机数生成的种子
        seed=None,
    ):
        """
        A decision tree model for regression or classification problems.

        Parameters
        ----------
        classifier : bool (default: True)
            Whether to treat target values as categorical (True) or
            continuous (False)
        max_depth: int (default: None)
            The depth at which to stop growing the tree. If None, grow the tree
            until all leaves are pure.
        n_feats : int (default: None)
            Specifies the number of features to sample on each split. If None,
            use all features on each split.
        criterion : str (default: 'entropy')
            The error criterion to use when calculating splits. When
            `classifier` is False, valid entries are {'mse'}. When `classifier`
            is True, valid entries are {'entropy', 'gini'}.
        seed : int (default: None)
            Seed for the random number generator
        """
        if seed:
            np.random.seed(seed)

        self.depth = 0
        self.root = None

        self.n_feats = n_feats
        self.criterion = criterion
        self.classifier = classifier
        self.max_depth = max_depth if max_depth else np.inf

        # 如果是回归器，却用了信息增益，报错
        if not classifier and criterion in ["gini", "entropy"]:
            raise ValueError(
                "{} is a valid criterion only when classifier = True.".format(criterion)
            )
        # 如果是分类器，却用了MSE，报错
        if classifier and criterion == "mse":
            raise ValueError("`mse` is a valid criterion only when classifier = False.")

    def fit(self, X, Y):
        """
        Trains a binary decision tree classifier.

        Parameters
        ----------
        X : numpy array of shape (N, M)
            The training data of N examples, each with M features
        Y : numpy array of shape (N,)
            An array of integer labels ranging between [0, n_classes-1] for
            each example in X if `self.classifier`=True else the set of target
            values for each example in X.
        """
        # 如果是分类器，则类别数为max(Y) + 1，因为Y里的类别号是从[0, n_classes-1]
        # 如果是回归器，则为None
        self.n_classes = max(Y) + 1 if self.classifier else None

        # 如果没有限制特征数，就用X的全部特征，
        # 否则，就取X特征数和限制特征数，二选一，取最小。
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])

        # 用_grow()生成树，返回的是一个根节点Node对象
        self.root = self._grow(X, Y)

    def predict(self, X):
        """
        Use the trained decision tree to classify or predict the examples in X.

        Parameters
        ----------
        X : numpy array of shape (N, M)
            The training data of N examples, each with M features

        Returns
        -------
        preds : numpy array of shape (N,)
            The integer class labels predicted for each example in X if
            classifier = True, otherwise the predicted target values.
        """
        # 对于输入的X中的每一个样本x，即每一行数据
        # 对x进行预测_traverse(x, self.root)
        # 传入的参数为该条样本，及树的根节点对象
        # self._traverse(x, self.root)即可获得x样本的leaf值。
        # 最终return的是一个数组，每个元素是该条样本对应的预测值。
        return np.array([self._traverse(x, self.root) for x in X])

    def predict_class_probs(self, X):
        """
        Use the trained decision tree to return the class probabilities for the
        examples in X.

        Parameters
        ----------
        X : numpy array of shape (N, M)
            The training data of N examples, each with M features

        Returns
        -------
        preds : numpy array of shape (N, n_classes)
            The class probabilities predicted for each example in X
        """
        assert self.classifier, "`predict_class_probs` undefined for classifier = False"
        # 同predict(self, X)，仅仅是增加了prob=True
        # 返回的是一个数组，每个元素是该条样本对应的概率数组。
        return np.array([self._traverse(x, self.root, prob=True) for x in X])

    def _grow(self, X, Y):
        # if all labels are the same, return a leaf
        """分裂到仅剩下同一种label时，返回叶子。
            1.如果是分类器，就计算概率数组；否则，就拿这唯一的值作为平均值。
            2.然后将数值存入叶子节点，并返回。
        """
        # 例如三分类问题，只有一种label，例如我们有6个样本，它们的label是Y=[2,2,2,2,2,2]
        if len(set(Y)) == 1:
            if self.classifier:
                # 如果是分类器，则生成一个prob数组，
                # 假如是3分类问题
                # 初始化prob=[0,0,0],
                # 由于Y里都是类别2，因此随便找一个元素出来，Y[0]=2
                # prob[2]=1，即把那个唯一的类别的概率标记为1
                # 最终prob=[0,0,1],
                prob = np.zeros(self.n_classes)
                prob[Y[0]] = 1.0
                # 之后prob=[0,0,1]
                # 如果是分类器，传给leaf的是一个概率数组prob=[0,0,1],,每个元素对应自己的概率
                return Leaf(prob)
            else:
                # 如果是回归器，则直接把唯一的那个数值target传给leaf，Y[0]=2
                return Leaf(Y[0])

        # if we have reached max_depth, return a leaf
        """分裂到最大深度限制时，返回叶子。 
            1.如果是分类器，就强行计算概率数组；否则，就强行计算平均值。
            2.然后将数值存入叶子节点，并返回。
        """
        if self.depth >= self.max_depth:
            if self.classifier:
                # 如果Y=[3,2,1,3,1],Y中最大的数字为3，因此Y中的索引值范围为0到3，即4个
                # v = np.bincount(Y, minlength=7)的结果是
                # v = [0,2,1,2,0,0,0]，表示Y中A类有0个，B类有2个，C类有1个，D类有2个，E/F/G类均为0个。
                # v = np.bincount(Y, minlength=1)的结果是
                # v = [0,2,1,2]，表示Y中A类有0个，B类有2个，C类有1个，D类有2个。
                # 即v的元素个数为max(max(Y)+1,minlength)，
                # 之所以要设置minlength，是保证最后统计的v的个数，刚好要对应类别个数，避免某些未出现的类别被遗漏
                # v数组的每个元素，表示该类对应的样本个数，再除以总样本数，就是每种样本的占比或概率
                # v = [0,0.4,0.2,0.4]
                v = np.bincount(Y, minlength=self.n_classes) / len(Y)
            else:
                v = np.mean(Y, axis=0)
            # 如果是分类器，每种样本的概率数组v = [0,0.4,0.2,0.4]传给叶子节点。
            # 如果是回归器，把这些样本的均值，即一个标量，发给叶子节点。
            return Leaf(v)

        # N为样本总量，M为特征数
        N, M = X.shape
        self.depth += 1
        # 在所有M个特征中，随机找n_feats次特征，采用无放回抽样（replace=False）。
        # 默认是有放回，replace = True，可以取无穷多次
        # 无放回，replace = False，可以取最多M次，因此self.n_feats必须小于M
        # feat_idxs 数组长度为self.n_feats，元素最大范围从0到M-1。
        # 例如 M = 5，n_feats = 3，
        # feat_idxs = [2,1,4],表示抽了三次，分别是第2/1/4特征，0/3特征未被抽到。
        feat_idxs = np.random.choice(M, self.n_feats, replace=False)

        # greedily select the best split according to `criterion`
        # feat 最优的分割特征的id
        # thresh 最优的分割特征的id下对应的最佳分割点值
        feat, thresh = self._segment(X, Y, feat_idxs)
        """根据最优分割特征feat，最优分割点值thresh，来划分左右子树。
            X[:, feat]=[1,4,3,5,3,1,8,8]
            thresh=2
            l =  [0 5]
            r =  [1 2 3 4 6 7]
            即np.argwhere()是把符合条件的索引返回来，flatten()即拉成一维数组
            l是≤thresh的值在X[:, feat]里的索引值列表
            r是＞thresh的值在X[:, feat]里的索引值列表
        """
        l = np.argwhere(X[:, feat] <= thresh).flatten()
        r = np.argwhere(X[:, feat] > thresh).flatten()

        # grow the children that result from the split
        """递归式生长
        _grow()：
        传入的是一堆样本的X和Y，
        返回的是一个分裂节点（记录着左节点、右节点、分割特征、分割点值）
        首次调用_grow()，等左右子树生长完了，即返回根节点。
        左子树、右子树的生长过程，又重复这样的递归。
        直到最终到达了叶子节点
        1.单侧子树只有一种label，即上面的 if len(set(Y)) == 1 来处理
        2.最大深度限制，即上面的 if self.depth >= self.max_depth 来处理
        """
        left = self._grow(X[l, :], Y[l])
        right = self._grow(X[r, :], Y[r])
        return Node(left, right, (feat, thresh))

    def _segment(self, X, Y, feat_idxs):
        """
        Find the optimal split rule (feature index and split threshold) for the
        data according to `self.criterion`.
        """
        best_gain = -np.inf
        split_idx, split_thresh = None, None
        for i in feat_idxs:
            """获取分割值的方法非常简单
            解释如下：
            假设某列特征的序列为vals=[1,4,3,5,3,1,8,8]
            np.unique的作用就是去重，同时排序：
            levels =  [1 3 4 5 8]
            然后分别准备一个去尾数组，一个去头数组：
            levels[:-1] =  [1 3 4 5]
            levels[1:] =  [3 4 5 8]
            两个数组的和，是对应元素的相加：
            levels[:-1] + levels[1:] =  [ 4  7  9 13]
            数组除以2，即对应元素除以2：
            thresholds =  [2.  3.5 4.5 6.5]
            核心思想是分割点正是levels里所有相邻元素的平均值（或者中位值）
            """
            vals = X[:, i]
            levels = np.unique(vals)
            thresholds = (levels[:-1] + levels[1:]) / 2

            """获取本特征i的所有分割增益
            假设某列特征的序列为vals = [1,4,3,5,3,1,8,8]
            t依次取从thresholds =  [2.  3.5 4.5 6.5]取出不同的分割点。
            即可获得特征id为i的特征对应的分割增益数组gains，每个元素为一个分割点的分割增益。
            gains里的最大值，就是本特征i的最佳表现。
            """
            gains = np.array([self._impurity_gain(Y, t, vals) for t in thresholds])

            """遍历feat_idxs里的每一个特征i
            最终保存分的最好的那个特征信息，即：
            最佳分割特征split_idx，
            最佳分割特征的最佳分割点split_thresh
            最佳分割特征的最佳分割点带来的增益best_gain
            
            """
            if gains.max() > best_gain:
                split_idx = i
                best_gain = gains.max()
                # 因为gains是依次迭代thresholds中的元素获得的数组，
                # 因此gains.argmax()正是最佳分割点的索引值。
                split_thresh = thresholds[gains.argmax()]

        return split_idx, split_thresh

    def _impurity_gain(self, Y, split_thresh, feat_values):
        """
        Compute the impurity gain associated with a given split.

        IG(split) = loss(parent) - weighted_avg[loss(left_child), loss(right_child)]
        """
        if self.criterion == "entropy":
            loss = entropy
        elif self.criterion == "gini":
            loss = gini
        elif self.criterion == "mse":
            loss = mse


        """获取父节点的信息量
           记住：信息量仅与Y有关，与X无关！
           因为信息量仅仅取决于label的丰富度，越丰富，信息量越大。
        """
        parent_loss = loss(Y)

        # generate split
        """分割左右子树
            feat_values = np.array([1, 4, 3, 5, 3, 1, 8, 8])
            split_thresh = 2
            raw=np.argwhere(feat_values <= split_thresh)
            left = np.argwhere(feat_values <= split_thresh).flatten()
            right = np.argwhere(feat_values > split_thresh).flatten()
            可见以下输出：
            raw =  [[0]
                    [5]]
            left =  [0 5]
            right =  [1 2 3 4 6 7]
            即np.argwhere()是把符合条件的索引返回来，flatten()即拉成一维数组
            left是≤split_thresh的值在feat_values里的索引值列表
            right是＞split_thresh的值在feat_values里的索引值列表
            索引值列表的长度也就是子树的size，下面会用到。
        """
        left = np.argwhere(feat_values <= split_thresh).flatten()
        right = np.argwhere(feat_values > split_thresh).flatten()

        if len(left) == 0 or len(right) == 0:
            return 0

        # compute the weighted avg. of the loss for the children
        n = len(Y)
        n_l, n_r = len(left), len(right)
        # 求左右子树的信息量，
        # 用Y[left]来获取左子树的label列表，
        # 用Y[right]来获取右子树的label列表，
        e_l, e_r = loss(Y[left]), loss(Y[right])
        # 本次分割的信息量=左子树的数量占比*左子树的信息量+右子树的数量占比*右子树的信息量。
        child_loss = (n_l / n) * e_l + (n_r / n) * e_r

        # impurity gain is difference in loss before vs. after split
        ig = parent_loss - child_loss
        return ig

    def _traverse(self, X, node, prob=False):
        """
        对于传入的待预测样本和根节点，该方法会不断递归，直到达到叶子节点，即返回该叶子节点的值,
        这个值可以是分类中的label（标量），也可以是概率数组。
        也可能是回归里的target（标量）。
        参见leaf的value说明
        """

        # 递归终止条件：如果传入的节点是叶子节点
        if isinstance(node, Leaf):
            if self.classifier:
                # 如果是分类器，就返回该叶子的value数组的最大值的索引
                # value数组长度为类别总数，每个元素代表每个类别的概率
                # 如果需要返回概率，就直接返回概率数组，例如v = [0,0.4,0.2,0.4]
                # 注意：由于node此时是Leaf类对象，因此node.value实际是leaf.value，
                return node.value if prob else node.value.argmax()
            else:
                # 如果是回归器，返回叶子节点的标量值（就是之前训练时计算的叶子节点里的均值）
                return node.value

        # 拿到节点对象里保存的特征索引node.feature，
        # X因为是一条样本，因此为一维数组，通过X[node.feature]取出该节点的分裂特征的值
        # 如果该值≤节点里的分裂点值，就继续遍历左子树（传入参数：样本、节点的左节点）
        if X[node.feature] <= node.threshold:
            return self._traverse(X, node.left, prob)
        else:
            # 否则，就遍历右子树（传入参数：样本、节点的左节点）
            return self._traverse(X, node.right, prob)

# 以下三个函数是计算loss的三个方法
def mse(y):
    """
    Mean squared error for decision tree (ie., mean) predictions
    """
    """对于回归问题，用mse来计算
      np.mean(y)是预测值，y是真实值的数组，
      均方误差正是求取所有样本 残差的平方 的均值。某一样本的残差正是该样本真实值减去模型预测值。
      这个mse越大，说明预测的越不准
      这个mse越小，说明预测得越准
      所以父节点的mse减去左右子树的mse，如果很大，说明这个分裂非常成功。
    """
    return np.mean((y - np.mean(y)) ** 2)


def entropy(y):
    """
    Entropy of a label sequence
    """
    """
    y = [3,2,1,3,1],A类=0，B类=1，C类=2，D类=3
    hist = [0,2,1,2]，表示Y中A类有0个，B类有2个，C类有1个，D类有2个。
    ps =  [0.  0.4 0.2 0.4],表示每个类对应的概率。
    
    信息熵 = -sum[ pi*log2（pi）]
    对于 ps =  [0.  0.4 0.2 0.4]
    entropy = -[0.4*log2(0.4)+0.2*log2(0.2)+0.4*log2(0.4)]
    or = -[0.4ln0.4+0.2ln0.2+0.4ln0.4]/ln2
    注意：是在所有类别上求和，也就是最多K个乘积项（设K为类别总数）
    
    [p * np.log2(p) for p in ps if p > 0]
    = [-0.5287712379549449, -0.46438561897747244, -0.5287712379549449]
    
    -np.sum(·)
    = 1.5219280948873621
    
    """
    hist = np.bincount(y)
    ps = hist / np.sum(hist)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])


def gini(y):
    """
    Gini impurity (local entropy) of a label sequence
    """
    """
        y = [3,2,1,3,1],A类=0，B类=1，C类=2，D类=3
        hist = [0,2,1,2]，表示Y中A类有0个，B类有2个，C类有1个，D类有2个。
        N=5
        gini=sum[pi*(1-pi)]=1-sum[pi**2]
        """
    hist = np.bincount(y)
    ps = hist / np.sum(hist)
    return 1-sum([i**2 for i in ps])
    # # 原作者的定义。
    # N = np.sum(hist)
    # # i / N 即为pi
    # return 1 - sum([(i / N) ** 2 for i in hist])
