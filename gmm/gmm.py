import numpy as np
from numpy.testing import assert_allclose

"""定义了一个高斯混合模型类



"""
class GMM(object):
    def __init__(self, C=3, seed=None):
        """
        A Gaussian mixture model trained via the expectation maximization
        algorithm.

        Parameters
        ----------
        C : int (default: 3)
            The number of clusters / mixture components in the GMM
        seed : int (default: None)
            Seed for the random number generator
        """
        self.C = C  # number of clusters
        self.N = None  # number of objects
        self.d = None  # dimension of each object

        if seed:
            np.random.seed(seed)

    def _initialize_params(self):
        """
        Randomly initialize the starting GMM parameters
        """

        """
        C=4   就是假设有4个高斯核
        N=10   10个样本
        d=2    每个样本为2维

        rr= [0.85827926 0.57980022 0.84407705 0.93940592]

        pi是分配给每个高斯的权重，4个高斯核，所以有4个元素
        pi =  [0.26641708 0.17997485 0.26200859 0.29159948]

        Q是每条样本属于不同高斯核的概率，10个样本，所以是10行；4个高斯核，所以有4列
        Q =  [[0. 0. 0. 0.]
             [0. 0. 0. 0.]
             [0. 0. 0. 0.]
             [0. 0. 0. 0.]
             [0. 0. 0. 0.]
             [0. 0. 0. 0.]
             [0. 0. 0. 0.]
             [0. 0. 0. 0.]
             [0. 0. 0. 0.]
             [0. 0. 0. 0.]]
        Q.shape =  (10, 4)

        mu是均值向量，4个高斯核，所以有4行；特征为2维，所以是2列
        mu =  [[-3.06807629  8.43418506]
             [-2.81441395 -3.46600049]
             [-3.54821313  5.67097503]
             [-0.64502243  2.7469061 ]]
        mu.shape =  (4, 2)

        sigma是协方差矩阵，4个高斯核，所以有4行；特征为2维，所以是d*d即2*2
        sigma =  [[[1. 0.]
              [0. 1.]]

             [[1. 0.]
              [0. 1.]]

             [[1. 0.]
              [0. 1.]]

             [[1. 0.]
              [0. 1.]]]
        sigma.shape =  (4, 2, 2)

        """

        C, d = self.C, self.d
        # 产生C个随机数。
        # 比如C=4，则rr= [0.85827926 0.57980022 0.84407705 0.93940592]
        rr = np.random.rand(C)

        self.pi = rr / rr.sum()  # cluster priors
        self.Q = np.zeros((self.N, C))  # variational distribution q(T)
        self.mu = np.random.uniform(-5, 10, C * d).reshape(C, d)  # cluster means
        self.sigma = np.array([np.identity(d) for _ in range(C)])  # cluster covariances
        # print("self.sigma  = ",self.sigma )

        self.best_pi = None
        self.best_mu = None
        self.best_sigma = None
        self.best_elbo = -np.inf

    def likelihood_lower_bound(self):
        """
        Compute the LLB under the current GMM parameters
        """
        # N个样本，C维特征
        N = self.N
        C = self.C

        eps = np.finfo(float).eps
        expec1, expec2 = 0.0, 0.0
        for i in range(N):
            x_i = self.X[i]

            for c in range(C):
                pi_k = self.pi[c]
                z_nk = self.Q[i, c]
                mu_k = self.mu[c, :]
                sigma_k = self.sigma[c, :, :]

                log_pi_k = np.log(pi_k + eps)
                log_p_x_i = log_gaussian_pdf(x_i, mu_k, sigma_k)
                prob = z_nk * (log_p_x_i + log_pi_k)

                expec1 += prob
                expec2 += z_nk * np.log(z_nk + eps)

        loss = expec1 - expec2
        return loss

    def fit(self, X, max_iter=100, tol=1e-3, verbose=False):
        """
        Fit the parameters of the GMM on some training data.

        Parameters
        ----------
        X : numpy array of shape (N, d)
            A collection of `N` training data points, each with dimension `d`
        max_iter : int (default: 100)
            The maximum number of EM updates to perform before terminating
            training
        tol : float (default 1e-3)
            The convergence tolerance. Training is terminated if the difference
            in VLB between the current and previous iteration is less than
            `tol`.
        verbose : bool (default: False)
            Whether to print the VLB at each training iteration.

        Returns
        -------
        success : 0 or -1
            Whether training terminated without incident (0) or one of the
            mixture components collapsed and training was halted prematurely
            (-1)
        """
        # X是训练数据，N为训练数据总数，d为训练数据的特征维度
        self.X = X
        self.N = X.shape[0]  # number of objects
        self.d = X.shape[1]  # dimension of each object

        self._initialize_params()
        prev_vlb = -np.inf

        for _iter in range(max_iter):
            try:
                self._E_step()
                self._M_step()
                vlb = self.likelihood_lower_bound()

                if verbose:
                    print("{}. Lower bound: {}".format(_iter + 1, vlb))

                converged = _iter > 0 and np.abs(vlb - prev_vlb) <= tol
                # vlb为空，或者 迭代收敛范围足够小了，就退出循环。
                if np.isnan(vlb) or converged:
                    break

                prev_vlb = vlb

                # retain best parameters across fits
                # 如果表现最佳，则保存这套参数为最佳参数。
                if vlb > self.best_elbo:
                    self.best_elbo = vlb
                    self.best_mu = self.mu
                    self.best_pi = self.pi
                    self.best_sigma = self.sigma

            # 如果出现异常，说明碰到了奇异矩阵，就返回-1
            # 原因出在_E_step()里需要计算log_gaussian_pdf()里
            # 涉及计算协方差矩阵sigma的逆矩阵，一旦sigma为奇异，将不可解。
            except np.linalg.LinAlgError:
                print("Singular matrix: components collapsed")
                return -1

        # 如果迭代中未有异常，说明顺利结束了。
        return 0

    def _E_step(self):
        for i in range(self.N):
            # 抽出一条样本x_i
            x_i = self.X[i, :]

            denom_vals = []
            for c in range(self.C):
                """ 遍历4个高斯核。
                
                C=4   就是假设有4个高斯核
                N=10   10个样本
                d=2    每个样本为2维
                
                pi是归一化后的rr，4个高斯核，所以有4个元素
                pi =  [0.26641708 0.17997485 0.26200859 0.29159948]

                Q是样本属于不同高斯核的概率，10个样本，所以是10行；4个高斯核，所以有4列
                Q.shape =  (10, 4)
        
                mu是均值向量，4个高斯核，所以有4行；特征为2维，所以是2列
                mu.shape =  (4, 2)
        
                sigma是协方差矩阵，4个高斯核，所以有4行；特征为2维，所以是N*N即2*2
                sigma.shape =  (4, 2, 2)
                
                """
                # pi_c 为对应一个高斯核的初始权重
                pi_c = self.pi[c]
                # mu_c 是对应一个高斯核的均值（包含d个均值分量）
                mu_c = self.mu[c, :]
                # sigma_c 是对应一个高斯核的协方差矩阵（d*d）
                sigma_c = self.sigma[c, :, :]

                log_pi_c = np.log(pi_c)
                log_p_x_i = log_gaussian_pdf(x_i, mu_c, sigma_c)

                # log N(X_i | mu_c, Sigma_c) + log pi_c
                denom_vals.append(log_p_x_i + log_pi_c)

            # log \sum_c exp{ log N(X_i | mu_c, Sigma_c) + log pi_c } ]
            """经历一轮C个高斯的概率计算，此时denom_vals数据存储着C个概率？
            假设 denom_vals =  [0.4 0.3 0.7 0.9]
            log_denom =  1.9899324102395517
            q_i =  [0.2039394  0.184532   0.27528939 0.33623922]
            """
            log_denom = logsumexp(denom_vals)
            # q_i 的每个元素 为 exp(denom_vals[i])-denom
            q_i = np.exp([num - log_denom for num in denom_vals])
            # 期望q_i数组里所有元素加起来等于1，检验一下。
            assert_allclose(np.sum(q_i), 1, err_msg="{}".format(np.sum(q_i)))

            # 返回本条样本属于4个高斯核的概率。
            self.Q[i, :] = q_i

    def _M_step(self):
        # 高斯个数C，特征维度N，样本集X
        C, N, X = self.C, self.N, self.X
        # 假设 Q =  [[0.2 0.3 0.4 0.1]
        #           [0.2 0.2 0.5 0.1]
        #           [0.1 0.1 0.6 0.2]]
        # 则对 Q 每列元素求和
        #  denoms =  [0.5 0.6 1.5 0.4]
        denoms = np.sum(self.Q, axis=0)

        # update cluster priors
        # 对denoms每个元素除以样本个数 N，假设 N=3,本应该列10个，但是我懒啊！
        # pi =  [0.16666667    0.2    0.5        0.13333333]
        self.pi = denoms / N

        # update cluster means
        # 假设 X =  [[2 3]
        #           [4 6]
        #           [5 3]]
        # 则拿 X 与 Q 中的每一列 分别做矩阵乘法，获得C个元素的数组，其中每个元素为数组。
        # nums_mu =  [array([1.7, 2.1]), array([1.9, 2.4]), array([5.8, 6. ]), array([1.6, 1.5])]
        nums_mu = [np.dot(self.Q[:, c], X) for c in range(C)]

        # 假设 mu =  [[-2.07900931  7.51359397]
        #            [-1.61742647 -3.99580112]
        #            [5.86325371 -1.63579457]
        #            [ 0.6632187  -0.33279188]]
        # 依次用以下元组中的第一个元素即数组，除以第二个元素
        # (num, den) =  (array([1.7, 2.1]), 0.5)
        # (num, den) =  (array([1.9, 2.4]), 0.6)
        # (num, den) =  (array([5.8, 6. ]), 1.5)
        # (num, den) =  (array([1.6, 1.5]), 0.4)
        # 那么
        # mu = [[3.4        4.2]
        #       [3.16666667 4.]
        #       [3.86666667 4.]
        #       [4.         3.75]]
        for ix, (num, den) in enumerate(zip(nums_mu, denoms)):
            self.mu[ix, :] = num / den if den > 0 else np.zeros_like(num)

        # update cluster covariances
        for c in range(C):
            mu_c = self.mu[c, :]
            n_c = denoms[c]

            outer = np.zeros((self.d, self.d))
            for i in range(N):
                wic = self.Q[i, c]
                xi = self.X[i, :]
                outer += wic * np.outer(xi - mu_c, xi - mu_c)

            outer = outer / n_c if n_c > 0 else outer
            self.sigma[c, :, :] = outer

        assert_allclose(np.sum(self.pi), 1, err_msg="{}".format(np.sum(self.pi)))


#######################################################################
#                                Utils                                #
#######################################################################

# 对多维高斯取对数的对数概率密度函数
def log_gaussian_pdf(x_i, mu, sigma):
    """
    Compute log N(x_i | mu, sigma)
    """
    """
    p(x)=[1/(2*pi)^(d/2)*det(sigma)^(1/2))]*exp[-(1/2)*(x-mu)^T*sigma^inverse*(x-mu)]
    log(p(x))=-(1/2)*d*log(2*pi)-(1/2)log(det(sigma))-(1/2)*(x-mu)^T*y
            其中sigma^inverse*(x-mu)=y,因为y是通过sigma*y=(x-mu)求出的。
    
    """
    # x_i,mu,sigma的行或列，三者的len相同，都是x的特征总数。
    n = len(mu)
    # np.log默认为自然对数
    # a=d*log(2*pi)
    a = n * np.log(2 * np.pi)


    # np.linalg.slogdet是获取矩阵的行列式的 符号sign，绝对值的自然对数值logdet
    # (sign, logdet) = np.linalg.slogdet(a)    sign=sign(-2),logdet=ln(|-2|)
    # 这里(-1, 0.69314718055994529) ，          det(a)=sign*exp(logdet)=-2
    # b = log(det(sigma))
    _, b = np.linalg.slogdet(sigma)

    # x = np.linalg.solve(a,b)
    # 即找到一个长度为N的一维数组x，使得ax=b，解出x。a是矩阵，b是一维数组，x是一维数组。
    # c=(x-mu)^T*y
    # np.linalg.solve求y时，需要计算sigma的逆矩阵，sigma为奇异矩阵将不可解。
    y = np.linalg.solve(sigma, x_i - mu)
    c = np.dot(x_i - mu, y)
    return -0.5 * (a + b + c)


def logsumexp(log_probs, axis=None):
    """
    Redefine scipy.special.logsumexp
    see: http://bayesjumping.net/log-sum-exp-trick/
    """
    """对对数概率log_probs，复原出真实概率np.exp(log_probs),然后求和，再取对数。
    其实仅仅是实现了
    用下面这一句就可以实现：
        return np.log(np.exp(log_probs).sum())
    但是防止数据下溢：数据接近0，就被约为0了。
    因此用了下面这个复杂的处理流程。其实就是通过提出一个最大值，变成+，因此避免了下溢。
    see: http://bayesjumping.net/log-sum-exp-trick/
    """
    # 例如log_probs =  [0.4 0.3 0.7 0.9]
    # _max =  0.9
    _max = np.max(log_probs)

    # ds =  [-0.5 -0.6 -0.2  0. ]
    ds = log_probs - _max

    # exp_sum =  2.9740730488846414
    exp_sum = np.exp(ds).sum(axis=axis)

    return _max + np.log(exp_sum)
