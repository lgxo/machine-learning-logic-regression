def create_data_naive_bayesian(m, is_naive):
    positive_mean = [0.5, 1]
    positive_rate = 0.5
    negative_mean = [-0.5, -1]
    negative_rate = 1 - positive_rate
    X = np.zeros((m, 2))
    y = np.zeros(m)
    cov_11 = 0.3
    cov_22 = 0.2
    if is_naive == 1:
        cov_12 = cov_21 = 0
    else:
        cov_12 = cov_21 = 0.1
    cov = [[cov_11, cov_12], [cov_21, cov_22]]      # X的两个维度的协方差矩阵
    positive_num = np.ceil(positive_rate * m).astype(np.int32)
    negative_num = np.ceil(negative_rate * m).astype(np.int32)
    X[:positive_num, :] = np.random.multivariate_normal(positive_mean, cov, size=positive_num)      # 根据协方差矩阵生成正态分布
    X[positive_num:, :] = np.random.multivariate_normal(negative_mean, cov, size=negative_num)      # 将正类反类区分开
    y[:positive_num] = 1
    y[positive_num:] = 0
    plt.scatter(X[:positive_num, 0], X[:positive_num, 1], c='r')
    plt.scatter(X[positive_num:, 0], X[positive_num:, 1], c='g')

    return X, y