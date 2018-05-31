# Lecture 11040A<Neural Networks>
## homework 2

Given: an 1-M-1 multi-layer perceptron (MLP) which input-ouput are denoted by x(t), y(t), t=1,2,...,N and the node function are f1(x)=tanh(x) for hidden layer, f2(x)=x for output layer.

+ Suppose there is a teacher signal, d(t), t=1,2,...,N, corresponding the input x(t). Please derive a backpropagation (BP) algorithm when a cost function is defined by:
  !(cost_function)[http://latex.codecogs.com]

+ Train the MLP using the BP algorithm in the case of
  M=2, N=1, x(1)=0.8, d(1)=0.72, and learning rate \mu=0.3
