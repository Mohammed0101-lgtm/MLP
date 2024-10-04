import numpy as np
import torch

# Block size used for batching data sequences
block_size = 8


class NeuralNet:
    """
    A simple feed-forward neural network class for training on numerical data.

    Attributes:
        layers_sizes (list of int): List containing the number of nodes in each layer.
        bias (bool): Whether to include a bias term in the computations.
        weights (list of np.array): List containing the weight matrices for each layer.
        lambda_r (float): Regularization parameter (lambda) for controlling overfitting.

    Methods:
        feed_forward(input_data): Performs forward propagation on the input data.
        initialize_weights(): Initializes the weight matrices using Xavier initialization.
        relu(val): Applies the ReLU activation function to the input.
        mean_squared_error(predicted, actual): Computes mean squared error between predicted and actual values.
        _backward(x, y): Performs backpropagation to calculate gradients.
        train(feature_mat, class_vec, iterations, learning_rate): Trains the neural network on a dataset.
        unroll_weights(rolled_data): Converts the weight matrices into a single flattened array.
        roll_weights(unrolled_data): Reconstructs weight matrices from a flattened array.
    """

    def __init__(self, layers_sizes, reg_lambda=0, bias=True):
        """
        Initializes the neural network.

        Args:
            layers_sizes (list of int): List of sizes of each layer.
            reg_lambda (float): Regularization lambda to prevent overfitting.
            bias (bool): Whether to use a bias in the layers.
        """
        self.n_layers = len(layers_sizes)
        self.layers_sizes = layers_sizes
        self.bias = bias
        self.weights = self.initialize_weights()  # Initialize weights for the network
        self.lambda_r = reg_lambda  # Regularization strength

    def feed_forward(self, input_data):
        """
        Performs forward propagation through the network.

        Args:
            input_data (np.array): Input data to the network.

        Returns:
            A (list): Activations at each layer.
            Z (list): Pre-activation values at each layer.
        """
        input_layer = input_data
        n_examples = input_layer.shape[0]

        # Lists to store activations (A) and pre-activations (Z) for each layer
        A = [None] * self.n_layers
        Z = [None] * self.n_layers

        # Forward propagation through the layers
        for layer_index in range(self.n_layers - 1):
            # Add bias to the input layer if required
            if self.bias:
                input_layer = np.concatenate(
                    (np.ones([n_examples, 1]), input_layer), axis=1)

            A[layer_index] = input_layer  # Store current activations
            # Get current layer's weights
            weight_matrix = self.weights[layer_index]

            # Print the shapes of the inputs and weights for debugging
            print(f"Layer {layer_index}:")
            print(f"Input shape: {input_layer.shape}")
            print(f"Weight shape: {weight_matrix.shape}")

            # Calculate pre-activation values (Z) and apply ReLU activation
            Z[layer_index + 1] = np.matmul(input_layer, weight_matrix.T)
            output_layer = self.relu(Z[layer_index + 1])
            input_layer = output_layer

        # Store final output layer activations
        A[self.n_layers - 1] = output_layer
        return A, Z

    def initialize_weights(self):
        """
        Initializes the weight matrices using Xavier initialization.

        Returns:
            weights (list of np.array): Initialized weight matrices for each layer.
        """
        weights = []
        next_layers_size = self.layers_sizes.copy()
        next_layers_size.pop(0)

        for layer_size, next_layer_size in zip(self.layers_sizes, next_layers_size):
            # Xavier initialization with small random values
            eps = np.sqrt(2.0 / (layer_size * next_layer_size))
            if self.bias:
                _tmp = eps * (np.random.randn(next_layer_size, layer_size + 1))
            else:
                _tmp = eps * (np.random.randn(next_layer_size, layer_size))
            weights.append(_tmp)

        return weights

    def relu(self, val):
        """
        Applies the ReLU activation function.

        Args:
            val (np.array): Input values.

        Returns:
            np.array: Output after applying ReLU.
        """
        return np.maximum(val, 0)

    def mean_squared_error(self, predicted, actual):
        """
        Calculates mean squared error between predicted and actual values.

        Args:
            predicted (np.array): Predicted output values.
            actual (np.array): Ground truth values.

        Returns:
            float: Mean squared error.
        """
        return np.mean((predicted - actual) ** 2)

    def _backward(self, x, y):
        """
        Performs backpropagation and computes gradients for weight updates.

        Args:
            x (np.array): Input data.
            y (np.array): Target output values.

        Returns:
            gradients (list of np.array): Computed gradients for each layer.
        """
        n_examples = x.shape[0]
        # Perform forward propagation to get activations
        A, Z = self.feed_forward(x)
        deltas = [None] * self.n_layers
        deltas[-1] = A[-1] - y  # Compute output error

        # Backpropagate the error for hidden layers
        for l_idx in np.arange(self.n_layers - 2, 0, -1):
            _tmp = self.weights[l_idx]

            if self.bias:
                # Remove bias if applicable
                _tmp = np.delete(_tmp, np.s_[0], 1)

            deltas[l_idx] = (np.matmul(_tmp.T, deltas[l_idx + 1].T)
                             ).T * (Z[l_idx] > 0)  # Apply ReLU derivative

        # Calculate gradients for each layer
        gradients = [None] * (self.n_layers - 1)
        for l_idx in range(self.n_layers - 1):
            grads_temp = np.matmul(deltas[l_idx + 1].T, A[l_idx])
            grads_temp = grads_temp / n_examples  # Average gradient over examples

            # Add regularization term to the gradients
            if self.bias:
                grads_temp[:, 1:] += (self.lambda_r /
                                      n_examples) * self.weights[l_idx][:, 1:]
            else:
                grads_temp += (self.lambda_r / n_examples) * \
                    self.weights[l_idx][:, 1:]

            gradients[l_idx] = grads_temp

        return gradients

    def train(self, feature_mat, class_vec, iterations=400, learning_rate=0.01):
        """
        Trains the neural network using gradient descent.

        Args:
            feature_mat (np.array): Input feature matrix.
            class_vec (np.array): Target output values.
            iterations (int): Number of training iterations.
            learning_rate (float): Learning rate for weight updates.
        """
        for _ in range(iterations):
            # Perform backpropagation to get gradients
            gradients = self._backward(feature_mat, class_vec)

            # Update weights for each layer
            for l_idx in range(self.n_layers - 1):
                self.weights[l_idx] -= learning_rate * gradients[l_idx]

            # Print loss every 100 iterations
            if _ % 100 == 0:
                predictions, _ = self.feed_forward(feature_mat)
                mse = self.mean_squared_error(predictions[-1], class_vec)
                print(f"Iteration {_}: MSE = {mse}")

    def unroll_weights(self, rolled_data):
        """
        Flattens weight matrices into a single array.

        Args:
            rolled_data (list of np.array): List of weight matrices.

        Returns:
            np.array: Flattened array of weights.
        """
        unrolled_array = np.array([])

        # Flatten and concatenate each weight matrix
        for one_layer in rolled_data:
            unrolled_array = np.concatenate(
                (unrolled_array, one_layer.flatten("F")))

        return unrolled_array

    def roll_weights(self, unrolled_data):
        """
        Reconstructs weight matrices from a flattened array.

        Args:
            unrolled_data (np.array): Flattened weight data.

        Returns:
            rolled_list (list of np.array): List of reconstructed weight matrices.
        """
        next_layers_sizes = self.layers_sizes.copy()
        next_layers_sizes.pop(0)
        rolled_list = []

        # Rebuild each layer's weight matrix
        extra_item = 1 if self.bias else 0
        for size_layer, next_layer_size in zip(self.layers_sizes, next_layers_sizes):
            n_weights = (next_layer_size * (size_layer + extra_item))
            data_tmp = unrolled_data[0: n_weights]
            data_tmp = data_tmp.reshape(
                next_layer_size, (size_layer + extra_item), order='F')
            rolled_list.append(data_tmp)
            unrolled_data = np.delete(unrolled_data, np.s_[0:n_weights])

        return rolled_list


def get_batch(sample):
    """
    Fetches a batch of data from either the training or validation set.

    Args:
        sample (str): 'train' to get data from the training set, 'val' for validation.

    Returns:
        x (torch.Tensor): Input batch.
        y (torch.Tensor): Target batch.
    """
    data = train_data if sample == 'train' else val_data
    index = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in index])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in index])

    return x, y


def get_data(filename):
    """
    Reads the data from a file, converts it to a tensor, and splits it into training and validation sets.

    Args:
        filename (str): The name of the file containing the data.

    Returns:
        train_data (torch.Tensor): Tensor containing the training data.
        val_data (torch.Tensor): Tensor containing the validation data.
    """
    with open(filename) as file:
        data = file.read()  # Read the file's contents

    # Convert characters to their corresponding ASCII values and then to a tensor
    data = torch.tensor([ord(c) for c in data])

    # Split the data into 90% for training and 10% for validation
    n = int(0.9 * len(data))
    train_data = data[:n]  # Training data (first 90%)
    val_data = data[n:]    # Validation data (last 10%)

    return train_data, val_data


if __name__ == '__main__':
    # Network and training parameters
    # Layer sizes for the neural network
    layers_sizes = [block_size, 128, 64, 1]
    epochs = 1000  # Number of training epochs
    block_size = 32  # Size of the input block (sequence length)
    batch_size = 64  # Batch size for training
    learning_rate = 0.001  # Learning rate for the optimizer
    reg_lambda = 0.001  # Regularization lambda to prevent overfitting

    # Load the training and validation data
    train_data, val_data = get_data('data.txt')

    # Create the neural network model
    model = NeuralNet(layers_sizes, reg_lambda=reg_lambda, bias=True)

    # Training loop
    for epoch in range(epochs):
        # Fetch a batch of training data
        x_batch, y_batch = get_batch('train')

        # Print the shape of the input and target batches for debugging
        print(f"x_batch shape: {x_batch.shape}")
        print(f"y_batch shape: {y_batch.shape}")

        # Perform a forward pass through the network
        output, _ = model.feed_forward(x_batch)

        # Compute the mean squared error loss
        loss = model.mean_squared_error(output[-1], y_batch)

        # Perform one iteration of training (backpropagation + weight updates)
        model.train(x_batch, y_batch, iterations=1,
                    learning_rate=learning_rate)

        # Print loss information every 100 epochs
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

    # Training completed
    print("Training complete.")
