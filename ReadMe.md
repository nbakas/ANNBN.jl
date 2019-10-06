# Fast & Accurate Artificial Neural Networks

> Scope

This is a Julia code for the implementation of ANNBN numerical scheme. It regards the computation of Artificial Neural Networks' weights, without a iterative training algorithm, by dividing the dataset in small neighborhoods. 

> Features of the Algorithm

- the training is vastly fast.
- it exhibits remarkably low errors.
- adheres to the underlying theory.
- same formulation for regression and classification. 
- high accuracy for computer vision tasks, without exploitation of spatial information.
- approximating highly nonlinear functions in multiple dimensions with low errors. 
- approximating higher-order partial derivatives.
- numerically solving Partial Differential Equations.
- the hyperparameters optimization is not necessary.
- the overfitting is inherently eliminated. The test-set errors are similar or smaller than the train errors.
- the results are exactly reproducible.
- the complexity of the proposed algorithm is of class P with O(mn^3) computing time.



> How to use

Download the files and run the examples comprised in the /test folder

EXAMPLES:: The root /test folder, contains the examples presented in the manuscript for regression and solution of PDEs. The subfolder /test/MNIST contains the ANNBN implementation for classification of the MNIST databse for computer vision. 

Each file create the variables xx_train (x in manuscript), yy_train (y), i_train (m), vars, (n), xx_test, yy_test, i_test (the coresponding, out-of-sample data for testing the prediction model). The weights for the hidden layer are stored in the variable a_all (Vector{Vector{Float64}}), and for the output layer in a_layer1 (with length nerons+1). You may change the number of neurons correspondngly.

The calculation of the neurons' wights is performed by using the functions ANNBN.train_layer_1_rbf, or ANNBN.train_layer_1_sigmoid_fast, depending on whether the radial basis (section 2.2 in manuscript) or the sigmoid (2.1.1-2.1.2) approach is utilized. Afterwards, the weights a_all, a_layer1 are utilized to predict for new observations, by using the ANNBN.predict_new_rbf or ANNBN.predict_new functions.

> Regression

Start with example in /test/n_variables.jl . You may change the generating function, or directly the input xx_train, yy_train, the number of observations, etc., and compare with other methods.

> Classification & Computer Vision

Start with example in /test/MNIST/__MNIST.jl . You may exactly reproduce the results in Table 2 of the manuscript. The same structure is appropriate for other classification problems.

