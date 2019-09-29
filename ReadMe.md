# Fast & Stable Artificial Neural Networks

> Scope

This is a Julia code for the implementation of ANNBN numerical scheme. It regards the computation of Artificial Neural Networks' weights, without a iterative training algorithm, by dividing the dataset in small neighborhoods. 

> Features of the Algorithm

- the training is vastly fast.
- it exhibits remarkably low errors.
- adheres to the underlying theory.
- same formulation for regression and classification. 
- high accuracy for cmputer vision tasks, without exploitation of spatial information.
- approximating highly nonlinear functions in multiple dimensions with low errors. 
- approximating higher-order partial derivatives.
- numerically solving Partial Differential Equations.
- the hyperparameters optimization is not necessary.
- the overfitting is inherently eliminated. The test-set errors are similar or smaller than the train errors.
- the results are exactly reproducible.
- the complexity of the proposed algorithm is of class P with O(mn^3) computing time.



> How to use

Download the files and run the examples comprised in the /test folder

EXAMPLES:: The examples presented in the manuscript are comprised in the folder __test__. It contins the examples presented in the manuscript for __regression__ and solution of __PDEs__. The subfolder __MNIST__ contains the ANNBN implementation for classification of MNIST databse for __computer vision__. 

They create the variables xx_train (x in manuscript), yy_train (y), i_train (m), vars, (n), xx_test, yy_test, i_test (the coresponding, out-of-sample data for testing the prediction model). The weights for the hidden layer are stored in a_all (Vector{Vector{Float64}}), and for the output layer at a_layer1 (with length nerons+1).

> Regression

Start with example in /test/100_variables.jl . You may change the input xx_train, yy_train, the number of observations, etc., and compare with other methods.

> Classification for Computer Vision

Start with example in /test/MNIST/__MNIST.jl . You may reproduce the results in Table 2 of the manuscript.

