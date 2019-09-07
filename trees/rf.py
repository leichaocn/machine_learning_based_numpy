import numpy as np
from .dt import DecisionTree

"""定义了一个随机森林类
fit(X, Y):训练n_trees个树，保存为一个树数组。
        即每次用bootstrap_sample(X, Y)生成新的样本集，训练一个树（通过调用DecisionTree类的实例化与fit()），并添加进树数组。
predict(X):
    1.通过DecisionTree类的_traverse()获取尺寸为(n_trees, N)的预测结果tree_preds，每一列，表示n_trees对一个样本的预测序列。
    2.通过_vote(tree_preds)获取一个长度为N的数组（N为X里的样本总数），该数组的每个元素为n_trees个树的结果的投票。
_vote(predictions):对每个样本的预测结果，用n_trees个树的结果进行投票。
"""
def bootstrap_sample(X, Y):
    # 总数据有N个样本，M个特征
    N, M = X.shape
    # replace=True，因此是有放回的抽样，理论上可以抽样无穷多次。
    # 第一个N 表示抽样范围，即在[0,N-1]的整数里抽样
    # 第二个N 表示抽样次数。
    # 假如N=5，抽5次，结果的idxs可能是[1,3,0,0,2]
    idxs = np.random.choice(N, N, replace=True)
    # 最后返回抽样的N个样本的X,Y。
    return X[idxs], Y[idxs]


class RandomForest:
    """
    A random forest of decision trees.
    """

    def __init__(
        self, n_trees, max_depth, n_feats, classifier=True, criterion="entropy"
    ):
        """
        An ensemble (forest) of decision trees where each split is calculated
        using a random subset of the features in the input.

        Parameters
        ----------
        n_trees : int
            The number of individual decision trees to use within the ensemble.
        max_depth: int or None
            The depth at which to stop growing each decision tree. If `None`,
            grow each tree until the leaf nodes are pure.
        n_feats : int
            The number of features to sample on each split.
        criterion : str (default: 'entropy')
            The error criterion to use when calculating splits. Valid entries
            are {'entropy', 'gini'}.
        """
        self.trees = []
        # 树的个数
        self.n_trees = n_trees
        # 每次分裂时的候选特征数目
        self.n_feats = n_feats
        # 每棵树的最大深度限制
        self.max_depth = max_depth
        # 指标类型
        self.criterion = criterion
        # 分类or回归
        self.classifier = classifier

    def fit(self, X, Y):
        """
        Create `n_trees`-worth of bootstrapped samples from the training data
        and use each to fit a separate decision tree.
        """
        self.trees = []
        for _ in range(self.n_trees):
            # 训练 n_trees 个树
            # 有放回地抽样出同样数量的样本
            X_samp, Y_samp = bootstrap_sample(X, Y)

            # 树的训练
            tree = DecisionTree(
                n_feats=self.n_feats,
                max_depth=self.max_depth,
                criterion=self.criterion,
                classifier=self.classifier,
            )
            tree.fit(X_samp, Y_samp)

            # 把这个树对象加入我们的树数组
            self.trees.append(tree)

    def predict(self, X):
        """
        Predict the target value for each entry in X.

        Parameters
        ----------
        X : numpy array of shape (N, M)
            The training data of N examples, each with M features

        Returns
        -------
        y_pred : np.array of shape (N,)
            Model predictions for each entry in X.
        """
        """
        self._traverse(x, self.root)即可获得x样本的leaf值。
        tree_preds是一个数组，
        每个元素又是一个数组[t._traverse(x, t.root) for x in X]，表示一颗树t对所有样本X的预测结果。
        t._traverse(x, t.root)表示一棵树t对该条样本x对应的预测值。
        
        因此tree_preds的实际尺寸为(n_trees, N)，
        即有n_trees行，每行是一个树；有N列，这个树对N个样本的预测结果
        """
        tree_preds = np.array([[t._traverse(x, t.root) for x in X] for t in self.trees])
        return self._vote(tree_preds)

    def _vote(self, predictions):
        """
        Return the aggregated prediction across all trees in the RF for each problem.

        Parameters
        ----------
        predictions : np.array of shape (n_trees, N)
            The array of predictions from each decision tree in the RF for each
            of the N problems in X

        Returns
        -------
        y_pred : np.array of shape (N,)
            If classifier is True, the class label predicted by the majority of
            the decision trees for each problem in X. If classifier is False,
            the average prediction across decision trees on each problem.
        """
        """
        predictions的尺寸是(n_trees, N)
        即有n_trees行，每行是一个树；有N列，这个树对N个样本的预测结果
        
        for x in predictions 表示挑出predictions每一行，即每个树对N个样本的预测x
        for x in predictions.T 表示挑出predictions每一列，即n_trees个树对一个样本的预测x
        假设我们有5颗树，n_trees=5，产生5种预测，用x表示为：
        x = [3,2,1,3,1],A树预测结果是3,B树预测结果是2,C树预测结果是1,D树预测结果是3,E树预测结果是1
        
        np.bincount(x) = [0,2,1,2]，统计了x中每种结果的个数，
        例如np.bincount(x)[2]=1,表示x里预测值2总共出现了1次。
        
        np.bincount(x).argmax()，就是拿到出现次数最多的那个结果。
        
        最终形成的out数组，
        每个元素是一个样本的预测，来自n_trees个树。
        共有N个元素，对应树组（即我们的RF）对N个样本的预测结果。
        """
        if self.classifier:
            # 如果是分类
            # np.bincount(x).argmax()是获取n_trees个树返回的预测值序列x中出现次数最多的结果。
            # 此处貌似不支持返回概率数组
            out = [np.bincount(x).argmax() for x in predictions.T]
        else:
            # 如果是回归，
            # np.mean(x)是计算n_trees个树返回的预测值序列x的均值。
            out = [np.mean(x) for x in predictions.T]
        return np.array(out)
