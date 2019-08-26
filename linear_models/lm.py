import sys
import numpy as np

sys.path.append("..")
from utils.testing import is_symmetric_positive_definite, is_number

"""这段代码实现了5种回归

LinearRegression:用解析方法求取系数。因为是解析法，不需要计算损失、梯度。
RidgeRegression:同线性回归的寻优方法，仅仅是在解析式中加入了一个正则系数的对角阵。
LogisticRegression：梯度下降法，需要求梯度。
BayesianLinearRegressionUnknownVariance
BayesianLinearRegressionKnownVariance
"""


class LinearRegression:
    """
    The simple linear regression model is

        y = bX + e  where e ~ N(0, sigma^2 * I)

    In probabilistic terms this corresponds to

        y - bX ~ N(0, sigma^2 * I)
        y | X, b ~ N(bX, sigma^2 * I)

    The loss for the model is simply the squared error between the model
    predictions and the true values:

        Loss = ||y - bX||^2

    The MLE for the model parameters b can be computed in closed form via the
    normal equation:

        b = (X^T X)^{-1} X^T y

    where (X^T X)^{-1} X^T is known as the pseudoinverse / Moore-Penrose
    inverse.
    """

    def __init__(self, fit_intercept=True):
        self.beta = None
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        # 伪逆矩阵
        pseudo_inverse = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)
        self.beta = np.dot(pseudo_inverse, y)

    def predict(self, X):
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        return np.dot(X, self.beta)


class RidgeRegression:
    """
    Ridge regression uses the same simple linear regression model but adds an
    additional penalty on the L2-norm of the coefficients to the loss function.
    This is sometimes known as Tikhonov regularization.

    In particular, the ridge model is still simply

        y = bX + e  where e ~ N(0, sigma^2 * I)

    except now the error for the model is calcualted as

        RidgeLoss = ||y - bX||^2 + alpha * ||b||^2

    The MLE for the model parameters b can be computed in closed form via the
    adjusted normal equation:

        b = (X^T X + alpha I)^{-1} X^T y

    where (X^T X + alpha I)^{-1} X^T is the pseudoinverse / Moore-Penrose
    inverse adjusted for the L2 penalty on the model coefficients.
    """

    def __init__(self, alpha=1, fit_intercept=True):
        """
        A ridge regression model fit via the normal equation.

        Parameters
        ----------
        alpha : float (default: 1)
            L2 regularization coefficient. Higher values correspond to larger
            penalty on the l2 norm of the model coefficients
        fit_intercept : bool (default: True)
            Whether to fit an additional intercept term in addition to the
            model coefficients
        """
        self.beta = None
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        # b = (X^T X + alpha I)^{-1} X^T y
        # 此处的A表示上面的alpha*I，
        # np.eye表示生成对角矩阵
        # 符号@在numpy中用于表示两个矩阵相乘
        A = self.alpha * np.eye(X.shape[1])
        pseudo_inverse = np.dot(np.linalg.inv(X.T @ X + A), X.T)
        self.beta = pseudo_inverse @ y

    def predict(self, X):
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        return np.dot(X, self.beta)

# 逻辑回归核心类
class LogisticRegression:
    def __init__(self, penalty="l2", gamma=0, fit_intercept=True):
        """
        A simple logistic regression model fit via gradient descent on the
        penalized negative log likelihood.

        Parameters
        ----------
        penalty : str (default: 'l2')
            The type of regularization penalty to apply on the coefficients
            `beta`. Valid entries are {'l2', 'l1'}.
        gamma : float in [0, 1] (default: 0)
            The regularization weight. Larger values correspond to larger
            regularization penalties, and a value of 0 indicates no penalty.
        fit_intercept : bool (default: True)
            Whether to fit an intercept term in addition to the coefficients in
            b. If True, the estimates for `beta` will have M+1 dimensions,
            where the first dimension corresponds to the intercept
        """
        err_msg = "penalty must be 'l1' or 'l2', but got: {}".format(penalty)
        assert penalty in ["l2", "l1"], err_msg
        self.beta = None
        self.gamma = gamma
        self.penalty = penalty
        self.fit_intercept = fit_intercept

    def fit(self, X, y, lr=0.01, tol=1e-7, max_iter=1e7):
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        # l_prev表示上一轮损失，初始时设置为无穷大
        l_prev = np.inf
        # 给一个尺寸同X特征总数的beta，用[0,1]之间的均匀分布
        self.beta = np.random.rand(X.shape[1])
        for _ in range(int(max_iter)):
            # 计算本轮系数beta下的预测值y_pred
            y_pred = sigmoid(np.dot(X, self.beta))
            # 计算本轮系数beta下的损失，并与上一轮损失进行比较
            loss = self._NLL(X, y, y_pred)

            # 训练终止条件
            # 一是靠迭代次数max_iter，
            # 二是靠损失减小量l_prev - loss
            if l_prev - loss < tol:
                return
            # 保存本轮损失为“上一轮”损失
            l_prev = loss

            # 更新参数，用的是梯度下降，而非SGD，因为是全部的数据喂进去了。
            self.beta -= lr * self._NLL_grad(X, y, y_pred)

    def _NLL(self, X, y, y_pred):
        """
        Penalized negative log likelihood of the targets under the current
        model.

            NLL = -1/N * (
                [sum_{i=0}^N y_i log(y_pred_i) + (1-y_i) log(1-y_pred_i)] -
                (gamma ||b||) / 2
            )

        解读：
        原始损失logloss=-sum[y_true*log(y_pred)+(1-y_true)log(1-y_pred)]
        显然，每个样本只有log(y_pred)，即y_true=1时；或log(1-y_pred)，即y_true=0时
        因为log(x)当x在（0,1）之间时处于（-inf,0），是严格单调增。

        乘以负号后，-log(x)当x在（0,1）之间处于（0，inf），是严格单调减。
        x越接近1，-log(x)越接近0.

        当某样本的y_true=1时，预测损失为-log(y_pred)，
        y_pred越接近1，-log(y_pred)越接近0，给总损失贡献的损失越小。
        当某样本的y_true=0时，预测损失为-log(1-y_pred)，
        y_pred越接近0，1-y_pred越接近1，-log(1-y_pred)越接近0，给总损失贡献的损失越小。

        然后把所有样本的损失加起来，就是上面的原始损失logloss

        而惩罚项(gamma ||b||) / 2，我们希望它越小越好。
        因此它越大，让损失也越大，那么加上它就能实现这个作用。
        NLL =1/N * [logloss + (gamma*||b||)/2]
        注意：1.至于1/N，我觉得意义不大。因为损失的目的是最小化，1/N不影响目的。
              2.||b||表示b的二阶范数的平方。
              3.logloss里的所有log，为了便于求梯度，假设为自然对数ln
              4.抽出负号到最左边，则(gamma*||b||)/2的系数就成了-1
              5.(gamma*||b||)/2就是下面的penalty，logloss就是下面的nll
        """
        # 此处略鸡肋，可用Y来替代。
        N, M = X.shape
        # 计算原生目标函数的损失
        nll = -np.log(y_pred[y == 1]).sum() - np.log(1 - y_pred[y == 0]).sum()
        # 计算惩罚项的损失
        order = 2 if self.penalty == "l2" else 1
        penalty = 0.5 * self.gamma * np.linalg.norm(self.beta, ord=order)** 2
        return (penalty + nll) / N

    def _NLL_grad(self, X, y, y_pred):
        """ Gradient of the penalized negative log likelihood wrt beta """
        """
        NLL =-1/N * {sum[y_true*ln(y_pred)+(1-y_true)ln(1-y_pred)] - (gamma*||b||)/2}
        先不管-1/N,
        1）求第一部分sum[·]对beta的导数。
        因为y_pred为sigmoid函数，因此：
        sum[y_true*1/y_pred*y_pred(1-y_pred)*x+(1-y_true)*(-1)/(1-y_pred)*y_pred(1-y_pred)*x] 
        =sum[y_true*(1-y_pred)*x-(1-y_true)*y_pred*x]
        =sum[(y_true-y_pred)*x]
        这里的y_true、y_pred、x均对应某一个特定样本。
        因此加和起来，就是求两个向量的内积np.dot(y - y_pred, X)
        2）求第二部分 -(gamma*||b||)/2对beta的导数。
        -1/2*gama*2*b=-gama*b
        即下面的d_penalty
        
        最后乘以1/N
        """
        N, M = X.shape
        p = self.penalty

        # gamma是惩罚系数，beta是惩罚项
        beta = self.beta
        gamma = self.gamma
        # l1norm（·）即求L1范数
        l1norm = lambda x: np.linalg.norm(x, 1)
        d_penalty = gamma * beta if p == "l2" else gamma * l1norm(beta) * np.sign(beta)
        # 原文如下：
        # return -(np.dot(y - y_pred, X) + d_penalty) / N
        # 应该为
        return -(np.dot(y - y_pred, X) + d_penalty) / N

    def predict(self, X):
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        return sigmoid(np.dot(X, self.beta))


class BayesianLinearRegressionUnknownVariance:
    """
    Bayesian Linear Regression
    --------------------------
    In its general form, Bayesian linear regression extends the simple linear
    regression model by introducing priors on model parameters b and/or the
    error variance sigma^2.

    The introduction of a prior allows us to quantify the uncertainty in our
    parameter estimates for b by replacing the MLE point estimate in simple
    linear regression with an entire posterior *distribution*, p(b | X, y,
    sigma), simply by applying Bayes rule:

        p(b | X, y) = [ p(y | X, b) * p(b | sigma) ] / p(y | X)

    We can also quantify the uncertainty in our predictions y* for some new
    data X* with the posterior predictive distribution:

        p(y* | X*, X, Y) = \int_{b} p(y* | X*, b) p(b | X, y) db

    Depending on the choice of prior it may be impossible to compute an
    analytic form for the posterior / posterior predictive distribution. In
    these cases, it is common to use approximations, either via MCMC or
    variational inference.

    Bayesian Regression w/ unknown variance
    ---------------------------------------
    If *both* b and the error variance sigma^2 are unknown, the conjugate prior
    for the Gaussian likelihood is the Normal-Gamma distribution (univariate
    likelihood) or the Normal-Inverse-Wishart distribution (multivariate
    likelihood).

        Univariate:
            b, sigma^2 ~ NG(b_mean, b_V, alpha, beta)

            sigma^2 ~ InverseGamma(alpha, beta)
            b | sigma^2 ~ N(b_mean, sigma^2 * b_V)

            where alpha, beta, b_V, and b_mean are parameters of the prior.

        Multivariate:
            b, Sigma ~ NIW(b_mean, lambda, Psi, rho)

            Sigma ~ N(b_mean, 1/lambda * Sigma)
            b | Sigma ~ W^{-1}(Psi, rho)

            where b_mean, lambda, Psi, and rho are parameters of the prior.

    Due to the conjugacy of the above priors with the Gaussian likelihood of
    the linear regression model we can compute the posterior distributions for
    the model parameters in closed form:

        B = (y - X b_mean)
        shape = N + alpha
        scale = (1 / shape) * {alpha * beta + B^T ([X b_V X^T + I])^{-1} B}

        sigma^2 | X, y ~ InverseGamma(shape, scale)

        A     = (b_V^{-1} + X^T X)^{-1}
        mu_b  = A b_V^{-1} b_mean + A X^T y
        cov_b = sigma^2 A

        b | X, y, sigma^2 ~ N(mu_b, cov_b)

    This allows us a closed form for the posterior predictive distribution as
    well:

        y* | X*, X, Y ~ N(X* mu_b, X* cov_b X*^T + I)
    """

    def __init__(self, alpha=1, beta=2, b_mean=0, b_V=None, fit_intercept=True):
        """
        Bayesian linear regression model with conjugate Normal-Gamma prior on b
        and sigma^2

            b, sigma^2 ~ NG(b_mean, b_V, alpha, beta)
            sigma^2 ~ InverseGamma(alpha, beta)
            b ~ N(b_mean, sigma^2 * b_V)

        Parameters
        ----------
        alpha : float (default: 1)
            The shape parameter for the Inverse-Gamma prior on sigma^2. Must be
            strictly greater than 0.
        beta : float (default: 1)
            The scale parameter for the Inverse-Gamma prior on sigma^2. Must be
            strictly greater than 0.
        b_mean : np.array of shape (M,) or float (default: 0)
            The mean of the Gaussian prior on b. If a float, assume b_mean is
            np.ones(M) * b_mean.
        b_V : np.array of shape (N, N) or np.array of shape (N,) or None
            A symmetric positive definite matrix that when multiplied
            element-wise by b_sigma^2 gives the covariance matrix for the
            Gaussian prior on b. If a list, assume b_V=diag(b_V). If None,
            assume b_V is the identity matrix.
        fit_intercept : bool (default: True)
            Whether to fit an intercept term in addition to the coefficients in
            b. If True, the estimates for b will have M+1 dimensions, where
            the first dimension corresponds to the intercept
        """
        # this is a placeholder until we know the dimensions of X
        b_V = 1.0 if b_V is None else b_V

        if isinstance(b_V, list):
            b_V = np.array(b_V)

        if isinstance(b_V, np.ndarray):
            if b_V.ndim == 1:
                b_V = np.diag(b_V)
            elif b_V.ndim == 2:
                assert is_symmetric_positive_definite(
                    b_V
                ), "b_V must be symmetric positive definite"

        self.b_V = b_V
        self.beta = beta
        self.alpha = alpha
        self.b_mean = b_mean
        self.fit_intercept = fit_intercept
        self.posterior = {"mu": None, "cov": None}
        self.posterior_predictive = {"mu": None, "cov": None}

    def fit(self, X, y):
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        N, M = X.shape
        beta = self.beta
        self.X, self.y = X, y

        if is_number(self.b_V):
            self.b_V *= np.eye(M)

        if is_number(self.b_mean):
            self.b_mean *= np.ones(M)

        # sigma
        I = np.eye(N)
        a = y - np.dot(X, self.b_mean)
        b = np.linalg.inv(np.dot(X, self.b_V).dot(X.T) + I)
        c = y - np.dot(X, self.b_mean)

        shape = N + self.alpha
        sigma = (1 / shape) * (self.alpha * beta ** 2 + np.dot(a, b).dot(c))
        scale = sigma ** 2

        # b_sigma is the mode of the inverse gamma prior on sigma^2
        b_sigma = scale / (shape - 1)

        # mean
        b_V_inv = np.linalg.inv(self.b_V)
        l = np.linalg.inv(b_V_inv + np.dot(X.T, X))
        r = np.dot(b_V_inv, self.b_mean) + np.dot(X.T, y)
        mu = np.dot(l, r)
        cov = l * b_sigma

        # posterior distribution for sigma^2 and c
        self.posterior = {
            "sigma**2": {"dist": "InvGamma", "shape": shape, "scale": scale},
            "b | sigma**2": {"dist": "Gaussian", "mu": mu, "cov": cov},
        }

    def predict(self, X):
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        I = np.eye(X.shape[0])
        mu = np.dot(X, self.posterior["b | sigma**2"]["mu"])
        cov = np.dot(X, self.posterior["b | sigma**2"]["cov"]).dot(X.T) + I

        # MAP estimate for y corresponds to the mean of the posterior
        # predictive
        self.posterior_predictive["mu"] = mu
        self.posterior_predictive["cov"] = cov
        return mu


class BayesianLinearRegressionKnownVariance:
    """
    Bayesian Linear Regression
    --------------------------
    In its general form, Bayesian linear regression extends the simple linear
    regression model by introducing priors on model parameters b and/or the
    error variance sigma^2.

    The introduction of a prior allows us to quantify the uncertainty in our
    parameter estimates for b by replacing the MLE point estimate in simple
    linear regression with an entire posterior *distribution*, p(b | X, y,
    sigma), simply by applying Bayes rule:

        p(b | X, y) = [ p(y | X, b) * p(b | sigma) ] / p(y | X)

    We can also quantify the uncertainty in our predictions y* for some new
    data X* with the posterior predictive distribution:

        p(y* | X*, X, Y) = \int_{b} p(y* | X*, b) p(b | X, y) db

    Depending on the choice of prior it may be impossible to compute an
    analytic form for the posterior / posterior predictive distribution. In
    these cases, it is common to use approximations, either via MCMC or
    variational inference.

    Bayesian linear regression with known variance
    ----------------------------------------------
    If we happen to already know the error variance sigma^2, the conjugate
    prior on b is Gaussian. A common parameterization is:

        b | sigma, b_V ~ N(b_mean, sigma^2 * b_V)

    where b_mean, sigma and b_V are hyperparameters. Ridge regression is a
    special case of this model where b_mean = 0, sigma = 1 and b_V = I (ie.,
    the prior on b is a zero-mean, unit covariance Gaussian).

    Due to the conjugacy of the above prior with the Gaussian likelihood in the
    linear regression model, we can compute the posterior distribution over the
    model parameters in closed form:

        A     = (b_V^{-1} + X^T X)^{-1}
        mu_b  = A b_V^{-1} b_mean + A X^T y
        cov_b = sigma^2 A

        b | X, y ~ N(mu_b, cov_b)

    which allows us a closed form for the posterior predictive distribution as
    well:

        y* | X*, X, Y ~ N(X* mu_b, X* cov_b X*^T + I)
    """

    def __init__(self, b_mean=0, b_sigma=1, b_V=None, fit_intercept=True):
        """
        Bayesian linear regression model with known error variance and
        conjugate Gaussian prior on b

            b | b_mean, sigma^2, b_V ~ N(b_mean, sigma^2 * b_V)

        Ridge regression is a special case of this model where b_mean = 0,
        sigma = 1 and b_V = I (ie., the prior on b is a zero-mean, unit
        covariance Gaussian).

        Parameters
        ----------
        b_mean : np.array of shape (M,) or float (default: 0)
            The mean of the Gaussian prior on b. If a float, assume b_mean is
            np.ones(M) * b_mean.
        b_sigma : float (default: 1)
            A scaling term for covariance of the Gaussian prior on b
        b_V : np.array of shape (N,N) or np.array of shape (N,) or None
            A symmetric positive definite matrix that when multiplied
            element-wise by b_sigma^2 gives the covariance matrix for the
            Gaussian prior on b. If a list, assume b_V=diag(b_V). If None,
            assume b_V is the identity matrix.
        fit_intercept : bool (default: True)
            Whether to fit an intercept term in addition to the coefficients in
            b. If True, the estimates for b will have M+1 dimensions, where
            the first dimension corresponds to the intercept
        """
        # this is a placeholder until we know the dimensions of X
        b_V = 1.0 if b_V is None else b_V

        if isinstance(b_V, list):
            b_V = np.array(b_V)

        if isinstance(b_V, np.ndarray):
            if b_V.ndim == 1:
                b_V = np.diag(b_V)
            elif b_V.ndim == 2:
                assert is_symmetric_positive_definite(
                    b_V
                ), "b_V must be symmetric positive definite"

        self.posterior = {}
        self.posterior_predictive = {}

        self.b_V = b_V
        self.b_mean = b_mean
        self.b_sigma = b_sigma
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        N, M = X.shape
        self.X, self.y = X, y

        if is_number(self.b_V):
            self.b_V *= np.eye(M)

        if is_number(self.b_mean):
            self.b_mean *= np.ones(M)

        b_V = self.b_V
        b_mean = self.b_mean
        b_sigma = self.b_sigma

        b_V_inv = np.linalg.inv(b_V)
        l = np.linalg.inv(b_V_inv + np.dot(X.T, X))
        r = np.dot(b_V_inv, b_mean) + np.dot(X.T, y)
        mu = np.dot(l, r)
        cov = l * b_sigma ** 2

        # posterior distribution over b conditioned on b_sigma
        self.posterior["b"] = {"dist": "Gaussian", "mu": mu, "cov": cov}

    def predict(self, X):
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        I = np.eye(X.shape[0])
        mu = np.dot(X, self.posterior["b"]["mu"])
        cov = np.dot(X, self.posterior["b"]["cov"]).dot(X.T) + I

        # MAP estimate for y corresponds to the mean of the posterior
        # predictive distribution
        self.posterior_predictive = {"dist": "Gaussian", "mu": mu, "cov": cov}
        return mu


#######################################################################
#                                Utils                                #
#######################################################################


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
