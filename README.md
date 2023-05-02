# Deep Neural Network in C with 2 layers
Class: COMP 605: Scientific Computing

Zack Humphries, Anuradha Agarwal, Thomas Keller


# 1. Files and folder in this directory:
- main.c: serial code for the deep neural network
- evaluation.h: contains all the functions used in the serial code 
- /openmp: contains all the files related to openMP
- /datasets: contains all the datasets used in this project
- /comparision: contains the forward propagation code used to compare results
- /CUDA: contains all the files related to CUDA

# 2. How to run serial code
There are 3 command line arguments: number of inputs(30 or 4800), learning rate, and number of epochs
- How to compile:
```
	gcc -o main main.c -lm
```
- How to run: 

```
	./main <numberofInputs> <learningRate> <numberofEpochs>
```

# 3. Dataset Used
https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data?resource=download

# 4. Compared to
https://github.com/nihil21/parallel_nn

# 5. Report
The file `final_project_report.pdf` contains an in-depth analysis of the paralleled neural networks and algorithms.

## Licence:
MIT
