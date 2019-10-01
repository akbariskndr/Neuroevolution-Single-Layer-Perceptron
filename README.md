# Neuroevolution for Single Layer Perceptron

## Introduction
This is my implementation of using Genetic Algorithm to seek for good hyperparameters for Single Layer Perceptron.
Method of using GA with ANN is called Neuroevolution.

The hyperparameters in this implementation are : 
1. Initial bias
2. Alpha (learning rate)
3. Theta (perceptron's activation function's threshold)

Those hyperparameters became the chromosome's genes.

The fitness function's value corresponds to the amount of epochs a perceptron required to be able to predict the targets correctly.
The lower the epochs the better the fitness value is.

## Modification
The implemented Single Layers Perceptron is able to do Linear Classification with any number of input neurons.

If you want to change the number of input neurons, set the ```neurons``` parameter on Perceptron or NeuroEvolution's constructor to the desired amount.
Also set the elements of ```data``` list with lists that have the same amount of elements as ```neurons``` parameter.

The default ```data``` and ```targets``` variables are set up to train perceptron to recognize three inputs AND boolean operator.

## Running
Just execute the file. ```python neuroevolution-perceptron.py```, easy peasy.  You could choose to run a single perceptron process by calling ```run_single_perceptron()``` method or the entire neuroevolution process by calling ```run_neuro_evolution()``` method.

## Dependency
Nothing

## Todo
- Modifying the fitness function to be able to consider the perceptron's accuracy.
- Using argv to change which method to run.
