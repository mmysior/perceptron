import pandas as pd
import numpy as np

def generate_data(m_examples, n_features, seed=None):
    """
    Generate a synthetic dataset with specified numbers of examples and features.

    This function creates a dataset consisting of random features and binary targets. 
    The features are generated from a uniform distribution, and the targets are binary, 
    randomly selected from {0, 1}. The random number generator can be seeded for 
    reproducibility.

    Parameters
    ----------
    m_examples : int
        The number of examples to generate in the dataset.
    n_features : int
        The number of features to generate for each example.
    seed : int, optional
        A seed for the random number generator to ensure reproducibility. 
        Default is None, which means no seed will be used.

    Returns
    -------
    data : pandas.DataFrame
        A DataFrame containing the generated features and targets. Each column represents 
        a feature named 'x0', 'x1', ..., 'x(n_features-1)', and an additional 'targets' 
        column for binary targets.

    Examples
    --------
    >>> generate_data(5, 3, seed=42)
       x0        x1        x2  targets
    0  0.374540  0.950714  0.731994        0
    1  0.598658  0.156019  0.155995        1
    2  0.058084  0.866176  0.601115        1
    3  0.708073  0.020584  0.969910        1
    4  0.832443  0.212339  0.181825        0

    Notes
    -----
    The function uses numpy's default random number generator and pandas to create the DataFrame.
    """
    generator = np.random.default_rng(seed)
    features = generator.random((m_examples, n_features))
    targets = generator.choice([0, 1], m_examples)
    data = pd.DataFrame(features, columns=[('x' + str(i)) for i in range(n_features)])
    data["targets"] = targets
    return data

def network_init(n_features, random_bias=False, seed=None):
    """
    Initialize the weights and bias for a single-layer neural network.

    This function generates a set of weights and a bias for a neural network layer with 
    a specified number of features. The weights are initialized randomly. The bias is 
    either set to zero or initialized randomly based on the `random_bias` parameter.
    A seed can be specified for the random number generator to ensure reproducibility.

    Parameters
    ----------
    n_features : int
        The number of features (inputs) for the neural network layer.
    random_bias : bool, optional
        If True, the bias is initialized randomly. Otherwise, it is set to 0.
        Default is False.
    seed : int or None, optional
        A seed for the random number generator to ensure reproducibility.
        If None, the generator is initialized without a fixed seed. 
        Default is None.

    Returns
    -------
    weights : ndarray
        A 1D array of shape (n_features,) containing the initialized weights.
    bias : float or ndarray
        The initialized bias. It is a float (0) if `random_bias` is False, 
        or a 1D array of shape (1,) with a random value if `random_bias` is True.

    Examples
    --------
    >>> network_init(3, random_bias=True, seed=42)
    (array([0.37454012, 0.95071431, 0.73199394]), array([0.59865848]))

    >>> network_init(2, random_bias=False, seed=42)
    (array([0.37454012, 0.95071431]), 0)

    Notes
    -----
    The function uses numpy's default random number generator for creating the weights
    and bias. The shape of the weights is a 1D array for ease of use in single-layer
    neural networks.
    """
    generator = np.random.default_rng(seed)
    weights = generator.random((1, n_features))[0]
    if random_bias:
        bias = generator.random((1, 1))[0]
    else:
        bias = 0
    return weights, bias

def get_weighted_sum(features, weights, bias):
    """
    Compute the weighted sum of features with the given weights and bias.

    This function calculates the dot product of the features and weights, and then adds 
    the bias to this product. It's a fundamental operation in many linear models and 
    neural networks, representing a linear combination of inputs.

    Parameters
    ----------
    features : ndarray
        An array of features (inputs). This can be a 1D array for a single set of features,
        or a 2D array for multiple sets, where each row represents a set of features.
    weights : ndarray
        An array of weights. The length of this array should match the number of features.
    bias : float or ndarray
        The bias term. Can be a scalar (if the same bias is to be added to all feature sets)
        or an array (if different biases are to be added to different feature sets).

    Returns
    -------
    ndarray
        The weighted sum of the features with the weights and bias. The shape of the return
        value depends on the input shapes. For a 1D `features` array and scalar `bias`, 
        the result is a scalar. For a 2D `features` array and scalar/vector `bias`, 
        the result is a 1D array.

    Examples
    --------
    >>> get_weighted_sum(np.array([1, 2, 3]), np.array([0.1, 0.2, 0.3]), 1)
    2.4

    >>> get_weighted_sum(np.array([[1, 2, 3], [4, 5, 6]]), np.array([0.1, 0.2, 0.3]), 1)
    array([2.4, 5.5])

    Notes
    -----
    The function is designed to work with NumPy arrays. Ensure that the dimensions of the
    input arrays (`features` and `weights`) match appropriately for matrix multiplication.
    """
    return np.dot(features, weights) + bias


def sigmoid(weighted_sum):
    """
    Compute the sigmoid function of the given weighted sum.

    This function applies the sigmoid activation function to a given input (weighted sum). 
    The sigmoid function is defined as 1 / (1 + exp(-x)), where x is the input. It is 
    commonly used in logistic regression and neural networks as an activation function 
    that maps any real-valued number into the range (0, 1).

    Parameters
    ----------
    weighted_sum : ndarray or scalar
        The weighted sum input to the sigmoid function. Can be a scalar, a 1D array, 
        or a 2D array.

    Returns
    -------
    ndarray or scalar
        The sigmoid of the input. The output shape is identical to the input shape.

    Examples
    --------
    >>> sigmoid(0)
    0.5

    >>> sigmoid(np.array([0, 2, -2]))
    array([0.5       , 0.88079708, 0.11920292])

    Notes
    -----
    The sigmoid function can lead to vanishing gradients when used in deep neural networks,
    as derivatives of very high or very low inputs are close to zero. It's also not zero-centered.
    """
    return 1 / (1 + np.exp(-weighted_sum))
