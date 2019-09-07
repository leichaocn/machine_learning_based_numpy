import numpy as np

#######################################################################
#                           Base Estimators                           #
#######################################################################


class ClassProbEstimator:
    def fit(self, X, y):
        self.class_prob = y.sum() / len(y)
        # 本质上就是下面这句啊！！
        # self.class_prob = np.mean(y)

    def predict(self, X):
        pred = np.empty(X.shape[0], dtype=np.float64)
        pred.fill(self.class_prob)
        return pred


class MeanBaseEstimator:
    def fit(self, X, y):
        self.avg = np.mean(y)

    def predict(self, X):
        pred = np.empty(X.shape[0], dtype=np.float64)
        pred.fill(self.avg)
        return pred


#######################################################################
#                           Loss Functions                            #
#######################################################################

# 回归任务的类
class MSELoss:
    def __call__(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)

    def base_estimator(self):
        return MeanBaseEstimator()

    def grad(self, y, y_pred):
        return -2 / len(y) * (y - y_pred)

    def line_search(self, y, y_pred, h_pred):
        # TODO: revise this
        Lp = np.sum((y - y_pred) * h_pred)
        Lpp = np.sum(h_pred * h_pred)

        # if we perfectly fit the residuals, use max step size
        return 1 if np.sum(Lpp) == 0 else Lp / Lpp

# 分类任务的损失类
class CrossEntropyLoss:
    def __call__(self, y, y_pred):
        eps = np.finfo(float).eps
        return -np.sum(y * np.log(y_pred + eps))

    def base_estimator(self):
        return ClassProbEstimator()

    def grad(self, y, y_pred):
        # 无穷小，为了防止分子为0.
        eps = np.finfo(float).eps
        return -y * 1 / (y_pred + eps)

    def line_search(self, y, y_pred, h_pred):
        raise NotImplementedError
