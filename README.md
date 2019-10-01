# Neuroevolution for Single Layer Perceptron

## Introduction
This is my implementation of using Genetic Algorithm to seek for good hyperparameters for Single Layer Perceptron.
Method of using GA with ANN is called Neuroevolution.

The hyperparameters in this implementation are : 
- Initial bias
- Alpha (learning rate)
- Theta (perceptron's activation function's threshold)
Those hyperparameters became the chromosome's genes.

The fitness function's value corresponds to the amount of epochs a perceptron required to be able to predict the targets correctly.
The lower the epochs the better the fitness value is.

## Modification
The implemented Single Layers Perceptron is able to do Linear Classification with any number of input neurons.

If you want to change the number of input neurons, set the ```neurons``` parameter on Perceptron or NeuroEvolution's constructor to the desired amount.
Also set the elements of ```data``` list with lists that have the same amount of elements as ```neurons``` parameter.

The default ```data``` and ```targets``` variables are set up to train perceptron to recognize three inputs AND boolean operator.

## Dependency
Nothing

## Todo
Modifying the fitness function to be able to consider the perceptron's accuracy.
