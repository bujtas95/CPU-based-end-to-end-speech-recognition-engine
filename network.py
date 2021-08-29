import layers 
import numpy as np

class Network:
    def __init__(self):
        self.jasper_blocks = []

    def add(self, jasper_block):
        self.jasper_blocks.append(jasper_block)

    # predict output for given input
    def predict(self, input_data):
        output = input_data
        for jasper in self.jasper_blocks:
            if jasper.tcs == False:
                a = 4
            if jasper.residual:
                temp_output = output
                temp_res_conv_output = float(0)
            for layer in jasper.inner_layers:
                if jasper.tcs == False:
                    output = layer.forward_propagation(output)
                elif layer.is_residualbranch:
                    if isinstance(layer, layers.ConvLayer):
                        temp_res_conv_output = layer.forward_propagation(temp_output)
                    elif isinstance(layer, layers.BNLayer):
                        res_bn_result = layer.forward_propagation(temp_res_conv_output)
                        output = np.add(res_bn_result, output)
                else:
                    output = layer.forward_propagation(output)
        return output