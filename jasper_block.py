import layers 

class Jasper_block:
    def __init__(self, encoderparams, residual, repeat, stride, dilation, tcs = True):
        self.inner_layers = []
        self.encoderparams = encoderparams
        self.residual = residual
        self.repeat = repeat
        self.stride = stride
        self.dilation = dilation
        self.tcs = tcs

        self.add_layers()

    def add_layers(self):        
        num_of_params = 7
        encoderparams = self.encoderparams
        residual = self.residual
        repeat = self.repeat
        stride = self.stride
        dilation = self.dilation

        for i in range(repeat):
            if self.tcs == False:
                self.inner_layers.append(layers.ConvLayer(weights=encoderparams[i*num_of_params + 0], stride=[1], dilation=dilation, depthwise=False, is_residualbranch = True))
                self.inner_layers.append(layers.BNLayer(gamma=encoderparams[i*num_of_params + 1], beta=encoderparams[i*num_of_params + 2], running_mean=encoderparams[i*num_of_params + 3], running_var=encoderparams[i * num_of_params + 4]))
                self.inner_layers.append(layers.ReluActivationLayer())
            else:
                self.inner_layers.append(layers.ConvLayer(weights=encoderparams[i*num_of_params + 0], stride=stride, dilation=dilation, depthwise=True))
                self.inner_layers.append(layers.ConvLayer(weights=encoderparams[i*num_of_params + 1], stride=[1], dilation=dilation, depthwise=False))
                self.inner_layers.append(layers.BNLayer(gamma=encoderparams[i*num_of_params + 2], beta=encoderparams[i*num_of_params + 3], running_mean=encoderparams[i*num_of_params + 4], running_var=encoderparams[i * num_of_params + 5]))
                if i == repeat-1 and residual:
                    self.inner_layers.append(layers.ConvLayer(weights=encoderparams[i*num_of_params + 7], stride=[1], dilation=dilation, depthwise=False, is_residualbranch=True))
                    self.inner_layers.append(layers.BNLayer(gamma=encoderparams[i*num_of_params + 8], beta=encoderparams[i*num_of_params + 9], running_mean=encoderparams[i*num_of_params + 10], running_var=encoderparams[i * num_of_params + 11], is_residualbranch=True))
                self.inner_layers.append(layers.ReluActivationLayer())