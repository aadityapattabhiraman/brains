## Loss Functions

### [Binary Cross Entropy](https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy.html#torch-nn-functional-binary-cross-entropy)

used primarily for binary classification tasks, where the goal is to classify input data into one of two possible classes  


### [Binary Cross Entropy with logits](https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html#torch.nn.functional.binary_cross_entropy_with_logits)

This is a slightly more efficient way of calculating the loss because it combines the sigmoid activation and the binary cross-entropy loss in one step, rather than applying the sigmoid activation function separately before computing the loss.  

### [Cross Entropy](https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html#torch.nn.functional.cross_entropy)

cross-entropy loss is used to measure the dissimilarity between the true label and the predicted probability distribution over all classes  

### [CTC](https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html#torch.nn.CTCLoss)

designed for sequence-to-sequence tasks, where the alignment between input and output sequences is unknown  

### [MAE](https://pytorch.org/docs/stable/generated/torch.nn.functional.l1_loss.html#torch.nn.functional.l1_loss)

measures the difference between the predicted values and the actual target values by taking the absolute differences. It's widely used in regression tasks where the goal is to predict continuous values  
L1 Loss (Mean Absolute Error) uses the absolute difference between the predicted and true values, making it more robust to outliers  

### [MSE](https://pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html#torch.nn.functional.mse_loss)

widely used loss function in regression tasks, where the goal is to predict continuous values  
L2 Loss (Mean Squared Error) uses the square of the difference, meaning that large errors are penalized more heavily. L2 loss is more sensitive to outliers  

### RMSE

measure the magnitude of the error in the same units as the target variable, making it easier to interpret compared to MSE  

