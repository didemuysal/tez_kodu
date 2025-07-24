from torch import optim

def get_optimizer(model_params, optimizer_name, lr):
    optimizer_name = optimizer_name.lower()

    if optimizer_name == 'adam':
        return optim.Adam(model_params, lr=lr)
    
    elif optimizer_name == 'adamw':
        return optim.AdamW(model_params, lr=lr, weight_decay=0.01)
    
    elif optimizer_name == 'sgd':
        return optim.SGD(model_params, lr=lr, momentum=0.9, nesterov=True)
    
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(model_params, lr=lr, alpha=0.99, eps=0.00000001)
    
    elif optimizer_name == 'adadelta':
        return optim.Adadelta(model_params, rho=0.9, eps=0.000001)
    
    else:
        raise ValueError("invalid optimizer")