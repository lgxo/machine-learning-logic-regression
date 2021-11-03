import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# number of samples
num_sample = 50
flag_naive = True
# newton descent
newton_accuracy = 1e-50
# gradient decent
alpha_gradient = 10
gradient_accuracy = 1e-5
# penalty item
flag_penalty = True
lambda_penalty = 1
# UCI
train_rate = 0.75


# append 1 to x_11, x_12 (x_11, x_12, 1)
def genXHat(x_mt):
    return np.append(x_mt, np.ones((x_mt.shape[0], 1)), axis=1)


# create samples
def creSamples(num, is_naive):
    pos_mean = [0.5, 1]
    pos_rate = 0.5
    neg_mean = [-0.5, -1]
    neg_rate = 1 - pos_rate
    x_mt = np.zeros((num, 2))
    y = np.zeros(num)
    cov_11 = 0.3
    cov_22 = 0.2
    if is_naive:
        cov_12 = cov_21 = 0
    else:
        cov_12 = cov_21 = 0.1
    # 协方差矩阵
    cov = [[cov_11, cov_12], [cov_21, cov_22]]
    pos_num = np.ceil(pos_rate * num).astype(np.int32)
    neg_num = np.ceil(neg_rate * num).astype(np.int32)
    # 根据协方差矩阵生成正态分布
    x_mt[:pos_num, :] = np.random.multivariate_normal(pos_mean, cov, size=pos_num)
    # 将正类反类区分开
    x_mt[pos_num:, :] = np.random.multivariate_normal(neg_mean, cov, size=neg_num)
    y[:pos_num] = 1
    y[pos_num:] = 0
    plt.scatter(x_mt[:pos_num, 0], x_mt[:pos_num, 1], c='blue', label="pos")
    plt.scatter(x_mt[pos_num:, 0], x_mt[pos_num:, 1], c='green', label="neg")
    return x_mt, y.reshape(num, 1)


# loss function
def lossFuc(x_hat_mt, y, beta_v, penalty_flag, penalty_lambda):
    obs_v = x_hat_mt@beta_v     # \beta^T \hat x_i
    tmp_v = -y*obs_v + np.log(1 + np.exp(obs_v))        # loss of each sample
    if penalty_flag:
        m = y.shape[0]
        # \frac{1}{m}\left(\sum_{i=1}^m{\left[ -y_i\beta^T\hat x_i + \ln (1 + e^{\beta^T\hat x_i}) \right]}
        # + \frac{\lambda}{2}\Vert \beta^T \Vert_2^2\right)
        result = (np.sum(tmp_v) + penalty_lambda*np.linalg.det(beta_v.T@beta_v)/2)/m
    else:
        # \sum_{i=1}^m{\left[ -y_i\beta^T\hat x_i + \ln (1 + e^{\beta^T\hat x_i}) \right]}
        result = np.sum(tmp_v)
    return result


# p_1(\hat x_i; \beta)
def posP(x_hat_mt, beta_v):
    exp_v = np.exp(x_hat_mt@beta_v)     # e^{\beta^T \hat x_i}
    return exp_v/(1 + exp_v)        # \frac{exp_v}{1 + exp_v}


# first derivative of loss function
def firstDerivative(x_hat_mt, y, beta_v, penalty_flag, penalty_lambda):
    if penalty_flag:
        # \frac{1}{m}\left(\hat x^T(p_1(\hat x;\beta)-y) + \lambda \beta\right)
        result = (x_hat_mt.T @ (posP(x_hat_mt, beta_v) - y) + penalty_lambda*beta_v) / y.shape[0]
    else:
        # \hat x^T(p_1(\hat x;\beta)-y)
        result = x_hat_mt.T@(posP(x_hat_mt, beta_v) - y)
    return result


# second derivative of loss function
def secondDerivative(x_hat_mt, beta_v, penalty_flag, penalty_lambda):
    pos_v = posP(x_hat_mt, beta_v)
    if penalty_flag:
        # \frac{1}{m}\left((\hat x*p_1(\hat x;\beta)*(1-p_1(\hat x;\beta)))^T\hat x + \lambda\right)
        result = ((x_hat_mt*pos_v*(1-pos_v)).T@x_hat_mt + penalty_lambda)/x_hat_mt.shape[0]
    else:
        # (\hat x*p_1(\hat x;\beta)*(1-p_1(\hat x;\beta)))^T\hat x
        result = (x_hat_mt*pos_v*(1-pos_v)).T@x_hat_mt
    return result


# newton descent
def newtonDescent(x_mt, y, penalty_newton=False, lambda_newton=0.):
    x_hat_mt = genXHat(x_mt)
    # initialize beta_v as [[0.][0.][0.]]
    beta_v = np.zeros(x_hat_mt.shape[1]).reshape(x_hat_mt.shape[1], 1)
    loss_pre = float("+inf")
    loss = lossFuc(x_hat_mt, y, beta_v, penalty_newton, lambda_newton)
    while loss_pre-loss > newton_accuracy:
        loss_pre = loss
        first_derivative = firstDerivative(x_hat_mt, y, beta_v, penalty_newton, lambda_newton)
        second_derivative = secondDerivative(x_hat_mt, beta_v, penalty_newton, lambda_newton)
        delta_v = np.linalg.inv(second_derivative)@first_derivative
        beta_v -= delta_v
        loss = lossFuc(x_hat_mt, y, beta_v, penalty_newton, lambda_newton)
        print(loss)
    return beta_v, loss


# gradient descent
def gradientDescent(x_mt, y, penalty_gradient=False, lambda_gradient=0.):
    alpha = alpha_gradient
    x_hat_mt = genXHat(x_mt)
    # initialize beta_v as [[1.][1.][1.]]
    beta_v = np.ones(x_hat_mt.shape[1]).reshape(x_hat_mt.shape[1], 1)
    loss_pre = float("+inf")
    # cycle condition
    loss = lossFuc(x_hat_mt, y, beta_v, penalty_gradient, lambda_gradient)
    while loss_pre-loss > gradient_accuracy:
        print("loss: ", end="")
        print(loss)
        loss_pre = loss
        beta_v_old = beta_v
        first_derivative = firstDerivative(x_hat_mt, y, beta_v, penalty_gradient, lambda_gradient)
        beta_v -= alpha*first_derivative
        loss = lossFuc(x_hat_mt, y, beta_v, penalty_gradient, lambda_gradient)
        if loss > loss_pre:
            beta_v = beta_v_old
            alpha /= 10
            print("alpha: ", end="")
            print(alpha)
            if alpha < 1e-99:
                print("too many samples")
                exit(0)
            loss_pre = float("+inf")
    return beta_v, loss_pre


def draw(beta_v, col, label):
    beta1, beta2, beta3 = beta_v[0][0], beta_v[1][0], beta_v[2][0]
    x_v = np.linspace(-2, 2, 50)
    if beta2 == 0:
        y_v = np.linspace(-2, 2, 50)
    else:
        y_v = (-beta1 * x_v - beta3) / beta2
    plt.plot(x_v, y_v, color=col, label=label)


# show results
def display(beta_v, loss, col, label):
    print("loss: ", end="")
    print(loss)
    print("beta_v:")
    print(beta_v)
    # draw line of gradient descent
    draw(beta_v, col, label)
    plt.legend(loc='upper left')


def testUCI():
    df = pd.read_csv('test.csv')
    mt = df.to_numpy()
    m, num_attribute = mt.shape
    position = np.around(train_rate * m).astype("int")

    x_mt = mt[..., :-3].astype("float")
    x_mt_train = x_mt[:position]
    x_mt_test = x_mt[position:]
    y_v = mt[..., -1].reshape(m, 1).astype("int")
    for i in range(m):
        if y_v[i] != 0:
            y_v[i] = 1
    y_v_train = y_v[:position]
    y_v_test = y_v[position:]
    beta_v, loss = newtonDescent(x_mt_train, y_v_train, flag_penalty, lambda_penalty)

    obs_v = genXHat(x_mt_test)@beta_v
    for i in range(y_v_test.shape[0]):
        if obs_v[i] < 0.5:
            obs_v[i] = 0
        else:
            obs_v[i] = 1
    obs_v = obs_v.astype("int")
    xor_v = obs_v ^ y_v_test

    zero_v = xor_v[obs_v == 0]
    rate = np.sum(zero_v.size)/y_v_test.shape[0]
    print("rate: ", end="")
    print(rate)


def main():
    x_mt, y = creSamples(num_sample, flag_naive)
    # print(x_mt)
    # print(y)

    # gradient descent
    # beta_v, loss = gradientDescent(x_mt, y, flag_penalty, lambda_penalty)
    # display(beta_v, loss, "red", "gradient")

    # newton descent
    beta_v, loss = newtonDescent(x_mt, y, flag_penalty, lambda_penalty)
    display(beta_v, loss, "red", "newton")

    plt.show()

    # testUCI()
    return


if __name__ == "__main__":
    main()
