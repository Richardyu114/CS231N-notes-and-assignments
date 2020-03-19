from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # softmax的损失函数是-log p(s_{yi}/sum(s_{yk})),但是在tf或者pytorch
    # 做DL时候似乎要乘以label,目前存疑，按照CS231N课程的笔记来做
    # 具体区别见：https://www.zhihu.com/question/341500352/answer/795497527
    # 有关softmax损失函数对W求导的推导过程见：
    # https://github.com/Richardyu114/CS231N-notes-and-assignments/blob/master/solutions/softmax%E6%B1%82%E5%AF%BC.jpg
    num_train = X.shape[0]
    num_classes = W.shape[1]
    # 仿照linear_svm.py的代码写
    for i in range(num_train):
        scores = np.dot(X[i,:], W)
        # 减去最大的值，防止梯度爆炸，不影响结果，详见lecture笔记
        scores -= np.max(scores)
        scores = np.exp(scores)
        scores /= np.sum(scores)
        loss += - np.log(scores[y[i]])
        # 根据矩阵求导结果得出循环下的方式
        for j in range(num_classes):
            if j != y[i]:
               dW[:, j] += X[i, :].T * scores[j]
            else:
               dW[:, j] += X[i, :].T * (scores[y[i]] - 1 )

    
    loss /= num_train
    loss += reg * np.sum(W * W)
    dW /= num_train
    dW += 2 * reg * W
        
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    #num_classes = W.shape[1]
    scores = np.dot(X ,W)  #N*C
    # 不加keepdims=True，得到的类似一个list
    scores_max = np.max(scores, axis=1, keepdims=True) #N*1
    scores -= scores_max #N*C
    scores = np.exp(scores) #N*C
    scores_sum = np.sum(scores, axis=1, keepdims=True) #N*1
    scores /= scores_sum
    # 不加np.sum得到的是数组
    loss += np.sum(- np.log(scores[range(num_train), list(y)]))
    dS = scores
    dS[range(num_train), list(y)] -= 1
    dW = np.dot(X.T, dS)
    loss = loss / num_train + reg * np.sum (W * W)
    dW = dW / num_train + 2 * reg * W



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
