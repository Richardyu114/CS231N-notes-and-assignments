from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                # !!!!! 根据lecture笔记， loss=sum(max(0,w_j*x_i-w_yi*x_i+delta))+reg*||W||^2
                dW[:, y[i]] += -X[i,:]
                dW[:, j] += X[i, :]
            

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    # !!!loss定义为所有样本的损失Loss的平均，那么梯度也要除以N
    # 详见 https://zh.d2l.ai/chapter_optimization/gd-sgd.html
    dW /= num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    # !!!更新正则项梯度
    dW += 2*reg*W # 逐元素相乘

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # line 39,40,41,47,48,51,52
    # 每次从X中去除一行，也就是1张图片和权重矩阵W相乘，得到一个length为C的array，每个索引位置代表该类别的score
    # 其中y[i]位置的score是GT的预测值，其他的都是异类的预测分数
    # 根据loss计算公式，X[i]是和W的每一列相乘得到一个类别的score，所以更新梯度时也是一列同时更新
    # !!!! 不要忘了正则项的梯度

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##################################################################
    # XW=S,f(S)=l,X(N,D), W(D,C),S(N,C)，l是标量，l=sum_{i}^{N}l_{i} #
    # 链式法则：l对W的偏导，等于l对S的偏导，再乘上S对W的偏导，结果是X.T #
    # 乘上l对S的偏导。S是score矩阵，每一行代表一个样本图片每类的预测分数#
    # 标量对矩阵的导数等于是逐元素求导，结果依旧是一个同维的矩阵        #      
    # l看作是一个N维列向量，每一维是S的一行与GT y比较得到的loss        #
    # 所以，l对S每个元素的导数可以进一步化简，然后根据合页损失函数得出  #
    # 最后的表达式                                                   #
    ##############################################################
    num_train = X.shape[0]
    #num_classes = W.shape[1]
    scores= np.dot(X, W)
    # 按照numpy的特性，y.shape是(N,),说明N是一维数组，N代表元素个数 #
    # 因此可以转换为list,这样就可以一次次得出包含正确类的score #
    # Nx1维
    correct_class_score = scores[range(num_train), list(y)].reshape(-1, 1)
    # broadcast机制，执行loss function
    scores = scores - correct_class_score + 1
    scores = np.where(scores<0, 0, scores)
    # 正确类别的位置也减去了，所以每行多加了delta,即1
    loss = (np.sum(scores)-num_train*1) / num_train
    loss += reg * np.sum(W * W)
    # 计算梯度
    # np.sign或许也可以实现类似功能
    dS = scores
    # 正确类别位置导数为-1的累加，累加次数是得分大于0且非正确类别的位置数
    a = np.sum(np.sign(scores), axis=1) - 1
    dS[range(num_train), list(y)] = -1 * a
    # 其他位置导数为1，0的位置没有导数
    dS = np.where(dS>0, 1 ,dS)
    dW = np.dot(X.T, dS)
    dW /= num_train
    dW += 2 *reg * W



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
