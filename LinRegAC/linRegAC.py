import numpy as np
from LinRegAC.utilsAC import getRmse

class LinReg:
    def __init__(self, lr=0.01, max_iter=1000, eps=1e-5, mode=None, stochGD=False):
        self.mode = mode
        self.lr = lr
        self.max_iter = max_iter
        self.w = None
        self.eps = eps
        self.stochGD = stochGD
        self.train_cost = []
        self.w_hists = []
        self.modes_dict = {1: 'using Normal equation w = (X^T X)^-1 X^T y', 2: 'using Gradient descent w = w - lr * grad, for overdetermined system', 3: 'using equation w = X^T (X X^T)^-1 y, for underdetermined system'}

    def fit(self, X, y, iter_step=1, w_hist=False):
        m, n = X.shape
        if m > n + 1:
            if self.mode == 1:
                self.__fit1(X, y)
            else:
                self.mode = 2
                self.__fit2(X, y, iter_step, w_hist)
        elif m < n + 1:
            self.mode = 3
            self.__fit3(X, y)
        else:
            raise ValueError('Unknown type of problem!')
        print(f'Fitting completed.\nMode = {self.mode}, {self.modes_dict[self.mode]}\n')

    def __fit1(self, X, y):
        '''
        Implementing Normal equation w = (X^T X)^-1 X^T y
        '''
        x_mat = np.append(np.ones((X.shape[0], 1)), X, axis=1)
        y_mat = np.array(y).reshape((len(y),1))
        if np.linalg.det(np.dot(x_mat.T, x_mat)) == 0:
            raise ValueError('Singular matrix, cannot find inverse. No solution for the given problem.')
        self.w = np.dot((np.linalg.inv(np.dot(x_mat.T, x_mat))), np.dot(x_mat.T, y_mat))
        
    def __fit2(self, X, y, iter_step=1, w_hist=False):
        '''
        Implementing Gradient descent w = w - lr * grad, for overdetermined system
        '''
        if self.mode != 2:
            raise ValueError('Mode not matching or not specified yet.')
        if self.stochGD:
            return self.__stochGradDesc(X, y, iter_step, w_hist)
        else:
            return self.__batchGradDesc(X, y, iter_step, w_hist)

    def __stochGradDesc(self, X, y, iter_step=1, w_hist=False):
        '''
        Implementing Stochastic Gradient descent
        '''
        print('Implementing Stochastic Gradient Descent.')
        n = len(X)
        feats = np.c_[np.ones(n), X]
        self.w = np.zeros(feats.shape[1])
        for _ in range(self.max_iter):
            w_new = self.w.copy()
            pred_all = np.dot(feats, w_new.T)
            self.train_cost.append(getRmse(pred_all, y))
            for i in range(n):
                pred_i = np.dot(feats[i], w_new.T)
                grad_i = self.__calc_grad(feats[i], y[i], pred_i) 
                w_new = w_new + np.dot(self.lr, grad_i) # w := w + lr * grad_i
            if (_ + 1) % iter_step == 0:
                print(f'Iteration {_ + 1:4d} | Loss = {getRmse(pred_all, y):.4f}')
            if self.__check_convergence(w_new):
                self.w = w_new
                if w_hist:
                    self.w_hists.append(self.w)
                print(f'Stopping criteria satisfied at iteration {_ + 1}.')
                break
            if w_hist:
                self.w_hists.append(self.w)
            self.w = w_new

    def __batchGradDesc(self, X, y, iter_step=1, w_hist=False):
        '''
        Implementing Batch Gradient descent
        '''
        print('Implementing Batch Gradient Descent.')
        n = len(X)
        feats = np.c_[np.ones(n), X]
        self.w = np.zeros(feats.shape[1])
        for _ in range(self.max_iter):
            pred = np.dot(feats, self.w.T)
            self.train_cost.append(getRmse(pred, y))
            grad = self.__calc_grad(feats, y, pred)
            w_new = self.w + np.dot(self.lr, grad) # w := w + lr * grad
            if (_ + 1) % iter_step == 0:
                print(f'Iteration {_ + 1:4d} | Loss = {getRmse(pred, y):.4f}')
            if self.__check_convergence(w_new):
                self.w = w_new
                if w_hist:
                    self.w_hists.append(self.w)
                print(f'Stopping criteria satisfied at iteration {_ + 1}.')
                break
            if w_hist:
                self.w_hists.append(self.w)
            self.w = w_new

    def __fit3(self, X, y):
        '''
        Implementing equation w = X^T (X X^T)^-1 y, for underdetermined system
        '''
        if self.mode != 3:
            raise ValueError('Mode not matching or not specified yet.')
        x_mat = np.append(np.ones((X.shape[0], 1)), X, axis=1)
        y_mat = np.array(y).reshape((len(y),1))
        if np.linalg.det(np.dot(x_mat, x_mat.T)) == 0:
            raise ValueError('Singular matrix, cannot find inverse. No solution for the given problem.')
        self.w = np.dot(x_mat.T, np.dot(np.linalg.inv(np.dot(x_mat, x_mat.T)), y_mat))

    def __calc_grad(self, X, y, pred): # grad = (y - pred) * X
        if type(pred) == np.int64 or type(pred) == np.float64:
            return np.dot(y - pred, X)
        return np.dot(y.to_numpy().flatten() - pred, X)
    
    def __check_convergence(self, w):
        return np.linalg.norm(w - self.w) < self.eps
    
    def predict(self, X):
        if self.mode == 1 or self.mode == 3:
            return self.__predict1(X)
        elif self.mode == 2:
            return self.__predict2(X)
        else:
            raise ValueError('Mode not matching or not specified yet.')
    
    def __predict1(self, X):
        '''
        Using Normal equation w = (X^T X)^-1 X^T y
        '''
        x_mat = np.append(np.ones((X.shape[0], 1)), X, axis=1)
        return np.dot(x_mat, self.w).flatten()

    def __predict2(self, X):
        '''
        Using Gradient descent w = w - lr * grad, for overdetermined system
        '''
        n = len(X)
        feats = np.c_[np.ones(n), X]
        pred = np.dot(feats, self.w.T)
        return pred
    
    def get_params(self):
        return self.w

    def get_train_cost(self):
        return self.train_cost
    
    def get_w_hists(self):
        return self.w_hists