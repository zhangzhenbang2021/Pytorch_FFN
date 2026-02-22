"""Utilities to configure PyTorch optimizers."""

import torch.optim as optim


def optimizer_from_args(model_parameters,
                        optimizer_name='adam',
                        learning_rate=0.001,
                        momentum=0.9,
                        adam_beta1=0.9,
                        adam_beta2=0.999,
                        epsilon=1e-8,
                        rmsprop_decay=0.9,
                        weight_decay=0.0):
    """Creates a PyTorch optimizer from arguments.

    Args:
        model_parameters: iterable of model parameters
        optimizer_name: one of 'sgd', 'momentum', 'adam', 'adagrad', 'rmsprop'
        learning_rate: initial learning rate
        momentum: momentum value (for momentum and rmsprop)
        adam_beta1: beta1 for Adam
        adam_beta2: beta2 for Adam
        epsilon: epsilon for Adam and RMSProp
        rmsprop_decay: alpha for RMSProp
        weight_decay: L2 regularization weight

    Returns:
        PyTorch optimizer instance
    """
    if optimizer_name == 'sgd':
        return optim.SGD(model_parameters, lr=learning_rate,
                         weight_decay=weight_decay)
    elif optimizer_name == 'momentum':
        return optim.SGD(model_parameters, lr=learning_rate,
                         momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        return optim.Adam(model_parameters, lr=learning_rate,
                          betas=(adam_beta1, adam_beta2), eps=epsilon,
                          weight_decay=weight_decay)
    elif optimizer_name == 'adagrad':
        return optim.Adagrad(model_parameters, lr=learning_rate,
                             weight_decay=weight_decay)
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(model_parameters, lr=learning_rate,
                             alpha=rmsprop_decay, momentum=momentum,
                             eps=epsilon, weight_decay=weight_decay)
    else:
        raise ValueError(f'Unknown optimizer: {optimizer_name}')


def build_lr_scheduler(optimizer, decay_factor=None, decay_steps=None):
    """Builds a learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer
        decay_factor: multiplicative decay factor
        decay_steps: steps between decays

    Returns:
        scheduler or None
    """
    if decay_factor is not None and decay_steps is not None:
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=decay_steps, gamma=decay_factor)
    return None
