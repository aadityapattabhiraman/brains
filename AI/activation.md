## Activation Functions

### [Sigmoid](https://pytorch.org/docs/stable/generated/torch.nn.functional.sigmoid.html#torch.nn.functional.sigmoid)

The Sigmoid function is a type of activation function commonly used in neural networks, especially in early models and certain types of output layers  

### [Softmax](https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html#torch.nn.functional.softmax)

Softmax function is an activation function commonly used in the output layer of neural networks, particularly for multi-class classification problems  

### [ReLU](https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html#torch.nn.functional.relu)

A simple activation function that outputs the input if it’s positive, or 0 if it’s negative  

### [Leaky ReLU](https://pytorch.org/docs/stable/generated/torch.nn.functional.leaky_relu.html#torch.nn.functional.leaky_relu)

Leaky ReLU allows small negative values for inputs less than zero, unlike standard ReLU which outputs 0 for negative inputs  
It helps avoid the "dying ReLU" problem and ensures that neurons can still contribute to learning, even if their outputs are negative  

### [ELU](https://pytorch.org/docs/stable/generated/torch.nn.functional.elu.html#torch.nn.functional.elu)

ELU is an activation function that improves upon ReLU by allowing negative values (as opposed to outputting 0 for negative inputs), which can help avoid the "dying ReLU" problem  
It has smoother gradients, which can lead to faster and more stable training, especially in deep networks  

### [GLU](https://pytorch.org/docs/stable/generated/torch.nn.functional.glu.html#torch.nn.functional.glu)

GLU stands for Gated Linear Unit, which is an activation function that is designed to improve performance in neural networks, particularly in tasks related to sequence modeling and natural language processing (NLP)  

## Optimizers

### [RMSPROP](https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop)

adaptive optimization algorithm that helps stabilize training, particularly when dealing with noisy or changing gradients. It is often used in deep learning applications and is effective in training models where the gradient magnitudes vary or are highly unstable, such as in recurrent neural networks (RNNs) and other complex deep learning architectures  

### [SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD)

Stochastic Gradient Descent (SGD) is a foundational optimization algorithm widely used in machine learning and deep learning. It is efficient and computationally feasible for large datasets, works well for online learning, and has the advantage of helping to escape local minima. However, its noisy updates can result in oscillations, and it may require careful tuning of the learning rate for effective training  

### [Adagrad](https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html#torch.optim.Adagrad)

Adagrad is an adaptive learning rate optimizer that adjusts the learning rate for each parameter based on its historical gradient information. This makes it highly effective for tasks with sparse data or when different parameters require varying step sizes. However, its main drawback is that the learning rate decays over time, which can lead to the optimizer taking very small steps later in training and potentially halting progress  

### [Adadelta](https://pytorch.org/docs/stable/generated/torch.optim.Adadelta.html#torch.optim.Adadelta)

Adadelta is used primarily in situations where adaptive learning rates are needed and where the manual tuning of the learning rate can be a burden. It's particularly popular in training deep learning models. It is often chosen when other algorithms like SGD or Adagrad may struggle due to decreasing learning rates or the complexity of the data  

### [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam)

it combines the advantages of both momentum and RMSprop while being computationally efficient and easy to use without manual learning rate tuning. It is widely used for a variety of tasks, from training deep networks to working with large, noisy datasets  
 
