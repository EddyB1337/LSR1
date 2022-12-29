# LSR1-TR Method for Deep Learning

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup with Google Colab](#setup-with-google-colab)
* [Setup via Terminal](#setup-via-terminal)
  * [Clone project](#clone-project)
  * [Setup with Linux](#setup-with-linux)
  * [Setup with Mac OS X](#setup-with-mac-os-x)
  * [Start the Code](#start-the-code)
  * [Output](#output)
* [Functions from torch](#functions-from-torch)
  * [Technical Functions](#technical-functions)
  * [Wolfe Conditions](#wolfe-conditions)
* [Functions from me](#functions-from-me)
* [Deep Learning Model (CNN)](#deep-learning-model-cnn)
* [Hyper-parameters](#hyper-parameters)
* [Sources](#sources)

## General info
This project includes the Limited Symmetric Rank 1 
Trust Region (LSR1-TR) optimizer. The algorithm 
was implemented using the Pytorch library. 
Some components were taken from the source code 
of the LBFGS implementation of Pytorch. 
Furthermore, we adopted a small deep learning model
to evaluate this algorithm. 
The algorithm is tested on the MNIST dataset. 
This dataset contains 28x28 images of handwritten 
digits and an associated label. 
This implementation can be used for any deep learning problem. 
It is also possible to switch to the GPU without any problems.

## Technologies
Project is created with:
* torch version : 1.13.0
* torchvision version : 0.14.0
* tqdm version : 4.64.1


The library torch is generally used for Deep Learning models,
therefore it is obvious to implement all functions in torch. 
Torch also provides interfaces for the use of a GPU by cuda. 
The MNIST data can be easily downloaded and used through 
torchvision. We use the library tqdm for aesthetic reasons 
to present a loading bar for the state of the epoch. 

## Setup with Google Colab
It is highly recommended to run the algorithm through 
Google Colab, we have created a corresponding notebook. 
This can be opened via the following link:

[Link to Notebook File](https://colab.research.google.com/drive/1yeEtvwcSainTdwEYk-4eus8si9TbCxIe?usp=sharing).

If Google Colab is not desired and this is to be run 
locally some steps are necessary. 

## Setup via Terminal

### Clone project
First, the project must be downloaded locally. 
To do this, we create a new folder via the terminal 
and also change to the path of the new folder. 
Open the terminal in your system and enter the 
following commands separately.
```
mkdir LSR1
cd LSR1
```
Clone the project using git.
```
git clone https://github.com/EddyB1337/LSR1.git
```
This ensures that you have the project locally on 
your computer. We still need to enter the folder path 
of the project.
```
cd LSR1
```
The further steps now depend on which operating system 
you are using. We offer a setup for OS X and Linux. 
For Windows users we recommend to download a virtual 
machine with a Linux version.

### Setup with Linux
Make sure you have python 3 installed. 
Use the following command to install the latest 
version of Python in a Linux system.
```
sudo apt-get install python3
```
Install the latest version of pip.
```
sudo apt install python3-pip
```
Finally, install all the necessary libraries 
using the requirements.txt file.
```
pip install -r requirements.txt
```
### Setup with Mac OS X
First install the package manager Homebrew.
```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
A prompt for a password will appear. 
Continue to follow the steps. 
After the successful installation of brew we can 
install python 3.
```
brew install python3
```
Now we can install pip as follows
```
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py
```
Finally, install all the necessary libraries 
using the requirements.txt file.
```
pip install -r requirements.txt
```

### Start the Code
The code is best started like this.
```
python3 main.py
```
With a favorite editor you can change the settings 
(hyperparameters and the size of the model) 
in the main.py with which you want to start the code. 
By default nano or vim is installed.
```
nano main.py
vim main.py
```
Of course it is also recommended to open and edit 
everything via an IDE like spyder, pycharm or similar. 
But we leave this to the user.

### Output
First, the MNIST dataset is downloaded. 
Then the loading bars are displayed. 
Then, for each epoch, a loading bar is given 
with the corresponding duration per step. 
After each epoch, a new loading bar appears 
and the accuracy of the test set and the loss 
of the training set and the test set of the epoch 
are run through.

## Functions from torch
We have taken some functions from torch.optim.LBFGS. 
This was well suited because there are many similarities. 
We have taken over functions that are mainly used for the 
technical part and two functions that are not part of the 
class but are still used, but are more related to 
the computational part. We have marked all functions 
we have taken over from torch.optim.LBFGS with a comment.

### Technical Functions

* def _numel(self):
* def _gather_flat_grad(self):
* def _add_grad(self, step_size, update):
* def _clone_param(self):
* def _set_param(self, params_data):
* def _directional_evaluate(self, closure, x, t, d):

These functions enable more technical tasks and did not 
play a significant part in the implementation.

### Wolfe Conditions

* def _cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
* def _strong_wolfe(obj_func,
                  x,
                  t,
                  d,
                  f,
                  g,
                  gtd,
                  c1=1e-4,
                  c2=0.9,
                  tolerance_change=1e-9,
                  max_ls=25):

These two functions performed an essential task in 
determining the learning rate. We have generously 
adopted them and would like to express our sincere 
thanks to the torch.optim.LBFGS creator.

## Functions from me

* def calculate_M(self, S, Y, gamma):
* def calculate_hess(self, Psi, M_inverse):
* def update_SY(self, s, y, old_s, old_y, cond_rest):
* def update_radius(self, r, trust_radius, s, T, rho):
* def trust_solver_OBS(self, M, P, lamb_gamma, trust_radius, gamma, flat_grad, psi):
* def trust_solver_cauchy(self, flat_grad, hess_1, hess_2, trust_radius):
* def trust_solver_steihaug(self, flat_grad, hess_1, hess_2, trust_radius):
* def step(self, closure):

### Calculate M Function
This function calculates the middle part without 
the inverse of the limited memory variant of the 
LSR1 matrix. Here S and Y are matrices whose columns 
represent the last m vectors s and y and gamma is the 
real value of the initial matrix B_0.

More detailed information can be found in the following source:
Richard, H., Byrd, & Jorge, Nocedal, & Robert, B., Schnabel. (1994). Representations of
quasi-Newton matrices and their use in limited memory methods, 147-149
Available from

https://link.springer.com/article/10.1007/BF01582063

### Calculate Hessian Function

### Update S and Y Function

### Update Radius Function

### Orthonormal Basis SR1 Solver Function

### Cauchy Point Calculation Solver Function

### Steihaug Conjugated Gradient Solver Function

### Step Function

## Deep Learning Model (CNN)
We have taken the CNN model from the following source.

https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118

This model consists of an input layer, a hidden layer 
and an end layer. We have changed the code so that the 
size of the dimensions of the individual layers can be 
passed as parameters. The user of the code has the 
possibility to change the size of the model in the 
main.py file. Of course, the user can create another model. 
The type of model is independent of the optimizer.

## Hyper-parameters

## Sources



