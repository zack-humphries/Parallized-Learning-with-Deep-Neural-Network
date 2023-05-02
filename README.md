# Deep Neural Network in C with 2 layers
## Class: COMP 605: Scientific Computing
## Names: Anuradha Agarwal, Thomas Keller, Zack Humphries 


# Files and folder in this directory:
- main.c: serial code for the deep neural network
- evaluation.h: contains all the functions used in the serial code 
- /openmp: contains all the files related to openMP
- /datasets: contains all the datasets used in this project
- /comparision: contains the forward propagation code used to compare results
- /CUDA: contains all the files related to CUDA

# How to run serial code
There are 3 command line arguments: number of inputs(30 or 4800), learning rate, and number of epochs
- How to compile:
```
	gcc -o main main.c -lm
```
- How to run: 

```
	./main <numberofInputs> <learningRate> <numberofEpochs>
```

# Dataset Used
https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data?resource=download

## Compared to
https://github.com/nihil21/parallel_nn

## Licence:
MIT
