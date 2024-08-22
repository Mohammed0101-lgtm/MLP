import numpy as np
import torch

block_size = 8

class NeuralNet():
    
    def __init__(self, layers_sizes, reg_lambda=0, bias=True):
        self.n_layers = len(layers_sizes)
        self.layers_sizes = layers_sizes
        self.bias = bias
        self.weights = self.initialize_weights()
        self.lambda_r = reg_lambda

    
    def feed_forward(self, input_data):
        input_layer = input_data
        n_examples = input_layer.shape[0]
        
        A = [None] * self.n_layers
        Z = [None] * self.n_layers

        for layer_index in range(self.n_layers - 1):
            if self.bias:
                input_layer = np.concatenate((np.ones([n_examples, 1]), input_layer), axis=1)
            
            A[layer_index] = input_layer
            weight_matrix = self.weights[layer_index]
            
            print(f"Layer {layer_index}:")
            print(f"Input shape: {input_layer.shape}")
            print(f"Weight shape: {weight_matrix.shape}")
            
            Z[layer_index + 1] = np.matmul(input_layer, weight_matrix.T)
            output_layer = self.relu(Z[layer_index + 1])
            input_layer = output_layer

        A[self.n_layers - 1] = output_layer
        
        return A, Z

    
    def initialize_weights(self):
        weights = []
        next_layers_size = self.layers_sizes.copy()
        next_layers_size.pop(0)

        for layer_size, next_layer_size in zip(self.layers_sizes, next_layers_size):
            eps = np.sqrt(2.0 / (layer_size * next_layer_size))
            if self.bias:
                _tmp = eps * (np.random.randn(next_layer_size, layer_size + 1))
            else:
                _tmp = eps * (np.random.randn(next_layer_size, layer_size))                    
            weights.append(_tmp)
        
        return weights

    
    def relu(self, val):
        return np.maximum(val, 0)

    
    def mean_squared_error(self, predicted, actual):
        return np.mean((predicted - actual) ** 2)

    
    def _backward(self, x, y):
        n_examples = x.shape[0]
        A, Z = self.feed_forward(x)
        deltas = [None] * self.n_layers
        deltas[-1] = A[-1] - y
    
        for l_idx in np.arange(self.n_layers - 2, 0, -1):
            _tmp = self.weights[l_idx]
            
            if self.bias:
                _tmp = np.delete(_tmp, np.s_[0], 1)
            
            deltas[l_idx] = (np.matmul(_tmp.T, deltas[l_idx + 1].T)).T * (Z[l_idx] > 0)

        gradients = [None] * (self.n_layers - 1)
        
        for l_idx in range(self.n_layers - 1):
            grads_temp = np.matmul(deltas[l_idx + 1].T, A[l_idx])
            grads_temp = grads_temp / n_examples
            
            if self.bias:
                grads_temp[:, 1:] += (self.lambda_r / n_examples) * self.weights[l_idx][:, 1:]
            else:
                grads_temp += (self.lambda_r / n_examples) * self.weights[l_idx][:, 1:]
        
            gradients[l_idx] = grads_temp
        
        return gradients

    
    def train(self, feature_mat, class_vec, iterations=400, learning_rate=0.01):
        for _ in range(iterations):
            gradients = self._backward(feature_mat, class_vec)
            
            for l_idx in range(self.n_layers - 1):
                self.weights[l_idx] -= learning_rate * gradients[l_idx]

            if _ % 100 == 0:  
                predictions, _ = self.feed_forward(feature_mat)
                mse = self.mean_squared_error(predictions[-1], class_vec)
                print(f"Iteration {_}: MSE = {mse}")

    
    def unroll_weights(self, rolled_data):
        unrolled_array = np.array([])

        for one_layer in rolled_data:
            unrolled_array = np.concatenate((unrolled_array, one_layer.flatten("F")))
        
        return unrolled_array


    def roll_weights(self, unrolled_data):
        next_layers_sizes = self.layers_sizes.copy()
        next_layers_sizes.pop(0)
        rolled_list = []

        extra_item = 1 if self.bias else 0
        for size_layer, next_layer_size in zip(self.layers_sizes, next_layers_sizes):
            n_weights = (next_layer_size * (size_layer + extra_item))
            data_tmp = unrolled_data[0: n_weights]
            data_tmp = data_tmp.reshape(next_layer_size, (size_layer + extra_item), order='F')
            rolled_list.append(data_tmp)
            unrolled_data = np.delete(unrolled_data, np.s_[0:n_weights])

        return rolled_list


def get_batch(sample):
    data = train_data if sample == 'train' else val_data
    index = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in index])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in index])
    return x, y


def get_data(filename):
    with open(filename) as file:
        data = file.read()

    data = torch.tensor([ord(c) for c in data])
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data


if __name__ == '__main__':
    layers_sizes = [block_size, 128, 64, 1]  
    epochs = 1000
    block_size = 32
    batch_size = 64
    learning_rate = 0.001
    reg_lambda = 0.001
    train_data, val_data = get_data('data.txt')
    model = NeuralNet(layers_sizes, reg_lambda=reg_lambda, bias=True)

    for epoch in range(epochs):
        x_batch, y_batch = get_batch('train')
        print(f"x_batch shape: {x_batch.shape}")
        print(f"y_batch shape: {y_batch.shape}")
        output, _ = model.feed_forward(x_batch)
        loss = model.mean_squared_error(output[-1], y_batch)

        model.train(x_batch, y_batch, iterations=1, learning_rate=learning_rate)

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

    print("Training complete.")
