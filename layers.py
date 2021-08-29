from layer import Layer
import numpy as np
from numpy import newaxis

class ConvLayer(Layer):
    def __init__(self, weights, stride, dilation, depthwise, is_residualbranch=False, bias=None):
        Layer.__init__(self, is_residualbranch)
        self.weights = weights
        self.stride = stride[0]
        self.dilation = dilation[0]
        self.depthwise = depthwise
        self.bias = bias

    def zero_pad(self, X, pad):
        X_pad = np.pad(X, ((0, 0), (0, 0), (pad, pad)), 'constant', constant_values=(0, 0))
        return X_pad

    def forward_propagation(self, input_data):
        (n_C_prev, n_H_prev, n_W_prev) = input_data.shape
        
        if self.dilation != 1 and self.depthwise:
            (w_cout, w_cin, w_kernel) = self.weights.shape
            w = np.zeros(( w_cout, w_cin, (w_kernel-1)*(self.dilation)+1), np.float32)
            i = 0
            dim = w.shape[0]
            for n in range(w_kernel):
                w[:dim,0,i] = self.weights[:,0,n]
                i += (self.dilation)
            self.weights = w

        (w_cout, w_cin, w_kernel) = self.weights.shape
        pad = w_kernel // 2

        n_W = int((n_W_prev - w_kernel + (2 * pad)) / (self.stride ))+1        

        Z = np.zeros(( w_cout, n_W), np.float32)

        if pad != 0:
            input_data = self.zero_pad(input_data, pad)
            
        if self.depthwise:
            Z  = np.array(list(map(lambda i: np.convolve(input_data[0, i,:], self.weights[i, 0, ::-1], 'valid')[::self.stride], range(w_cout))))
            Z = Z[:,newaxis,:]
        elif self.is_residualbranch:
            Z  = np.dot(self.weights[:, :, 0], input_data[0, :,:])[::self.stride]
            Z = Z[newaxis,:,:]
        else:
            Z  = np.dot(self.weights[:,:,0], input_data[:,0,:])[::self.stride]
            Z = Z[newaxis,:,:]

        if self.bias is not None:
            for i in range(len(self.bias)):
                Z[0,i,:] = Z[0,i,:] + self.bias[i]
        return Z

class ReluActivationLayer(Layer):
    def forward_propagation(self, input_data):
        ( n_C_prev, n_H_prev, n_W_prev) = input_data.shape
        Z = np.zeros((n_C_prev, n_H_prev, n_W_prev), np.float32)
        for i in range(n_H_prev):
            Z[0,i]  = np.maximum(0,input_data[0,i,:])
        return Z

class SoftmaxLayer(Layer):
    def forward_propagation(self, input_data):
        input_data = np.squeeze(input_data)
        e_x = np.exp(input_data - np.max(input_data))
        return np.transpose(np.log(e_x / e_x.sum(axis=0))) 

class BNLayer(Layer):
    def __init__(self, gamma, beta, running_mean, running_var, is_residualbranch=False):
        Layer.__init__(self, is_residualbranch)
        self.gamma = gamma
        self.beta = beta
        self.running_mean = running_mean
        self.running_var = running_var

    def forward_propagation(self, input_data):
        eps = 0.001
        (n_C_prev, n_H_prev, n_W_prev) = input_data.shape

        Z = np.zeros((n_C_prev, n_H_prev, n_W_prev), np.float32)

        for i in range(n_H_prev):
            x_norm = (input_data[0,i,:] - self.running_mean[i]) / np.sqrt(self.running_var[i] + eps)
            Z[0,i]  = self.gamma[i] * x_norm + self.beta[i]
        return Z
    
def ctc_decoder(input_data, labels):
    blank_id = len(labels)
    hypotheses = []
    labels_map = dict([(i, labels[i]) for i in range(len(labels))])
    decoded_prediction = []
    previous = len(labels)
    for p in input_data:
        if (p != previous or previous == blank_id) and p != blank_id:
            decoded_prediction.append(p)
        previous = p
    hypothesis = ''.join([labels_map[c] for c in decoded_prediction])
    hypotheses.append(hypothesis)
    return hypotheses