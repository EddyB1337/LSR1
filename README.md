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
  * [Calculate M Function](#calculate-m-function)
  * [Calculate Hessian Function](#calculate-hessian-function)
  * [Update S and Y Function](#update-s-and-y-function)
  * [Update Radius Function](#update-radius-function)
  * [Orthonormal Basis SR1 Solver Function](#orthonormal-basis-sr1-solver-function)
  * [Cauchy Point Calculation Solver Function](#cauchy-point-calculation-solver-function)
  * [Steihaug Conjugated Gradient Solver Function](#steihaug-conjugated-gradient-solver-function)
  * [Step Function](#step-function)
* [Deep Learning Model (CNN)](#deep-learning-model-cnn)
* [Loss Function](#loss-function)
* [Hyper-parameters](#hyper-parameters)
  * [lr and line_search_fn](#lr-and-line_search_fn)
  * [mu, nu and alpha_S](#mu-nu-and-alpha_s)
  * [trust_solver, newton_maxit and cg_iter](#trust_solver-newton_maxit-and-cg_iter)
  * [memory size](#memory-size)
  * [tr_radius](#tr_radius)
  * [tolerance_change and tolerance_grad](#tolerance_change-and-tolerance_grad)
* [Potential problems](#potential-problems)

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

The runtime should be changed to a GPU. Go to 
Runtime->Change runtime type 
and then select GPU to Hardware accelerator. 
If the file is in German, then 
Laufzeit->Laufzeittyp ändern
and then select GPU on Hardwarebeschleuniger.

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
Finally, install all necessary libraries 
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
Finally, install all necessary libraries 
using the requirements.txt file.
```
pip install -r requirements.txt
```

### Start the Code
The code is best started like this.
```
python3 main.py
```
With favorite editor you can change the settings 
(hyperparameters and the size of the model) 
in the main.py with which you want to start the code. 
By default nano or vim is installed.
```
nano main.py
vim main.py
```
Of course it is also recommended to open and edit 
everything via an IDE like spyder, pycharm or similar.

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
To understand these functions, the following sources should be consulted:
* Jorge, Nocedal, & Stephen, J., Wright. (2006). Numerical Optimization. 71-73, 146
* Joshua, D., Griffin, & Majid, Jahani, & Martin, Takáč, & Seyedalireza, Yektamaram, &
Wenwen Zhou. (2022). A minibatch stochastic Quasi-Newton method adapted for nonconvex
deep learning problems
* Richard, H., Byrd, & Jorge, Nocedal, & Robert, B., Schnabel. (1994). Representations of
quasi-Newton matrices and their use in limited memory methods, 147-149
* Oleg, Burdakov, & Yu-Hong, Dai, & Na, Huang. (2019). Stabilizied Barzilai-Borwein
* Johannes, Brust, & Jennifer, B., Erway, & Roummel, F., Marcia. (2016). On solving L-SR1
Trust-Region Subproblems

The following bullet points are functions that we have written.

* def calculate_M(self, S, Y, gamma):
* def calculate_hess(self, Psi, M_inverse):
* def update_SY(self, s, y, old_s, old_y, cond_rest):
* def update_radius(self, r, trust_radius, s, T, rho):
* def trust_solver_OBS(self, M, P, lamb_gamma, trust_radius, gamma, flat_grad, psi):
* def trust_solver_cauchy(self, flat_grad, hess_1, hess_2, trust_radius):
* def trust_solver_steihaug(self, flat_grad, hess_1, hess_2, trust_radius):
* def step(self, closure):

We discuss the functions in the appropriate order:

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
This function returns an orthogonal matrix P and a 
square diagonal matrix. To begin, we compute a reduced 
QR decomposition of Psi = Y - gamma * S. We take Psi 
from the previous function where we compute M. 
From the R of the QR decomposition, we compute a 
spectral decomposition of R@M^{-1}@R^T = U@L@U^T, where 
M^{-1} is the inverse of M. P=Q@U and L is returned. 
Note that because of the torch function, an imaginary 
part of 0 is always added, so we take only the real part.

More detailed information can be found in the following source:
Johannes, Brust, & Jennifer, B., Erway, & Roummel, F., Marcia. (2016). On solving L-SR1
Trust-Region Subproblems 
Available from 

https://arxiv.org/abs/1506.07222

### Update S and Y Function
This function updates the lists of the last m 
(memory size) vectors s and y. 
We have named these lists old_s and old_y. 
First, we check if the LSR1 update is well-defined, and 
if it is, we insert the current vectors s and y. 
If the memory size is already reached, we remove 
the first vector from both lists.

### Update Radius Function
This function updates the Trust Region Radius. 
We have taken the function from the following source.

Joshua, D., Griffin, & Majid, Jahani, & Martin, Takáč, & Seyedalireza, Yektamaram, &
Wenwen Zhou. (2022). A minibatch stochastic Quasi-Newton method adapted for nonconvex
deep learning problems
Available from 

https://optimization-online.org/2022/01/8760/

We have made a correction in the first if condition:
```
if rho < 0.5:
```
Furthermore, we have changed the first line:
```
rho = 0.5 * T * rho - r
```

### Orthonormal Basis SR1 Solver Function
We have taken the implementation of the function from 
the following source.

https://github.com/MATHinDL/sL_QN_TR

Keep in mind that we have converted Matlab code to Python Code.
This function solves a trust region subproblem using the 
LSR1 matrix. For a detailed description of this function, 
the following source is helpful.

Johannes, Brust, & Jennifer, 
B., Erway, & Roummel, F., Marcia. (2016). 
On solving L-SR1 Trust-Region Subproblems 
Available from

https://arxiv.org/abs/1506.07222

This solver has six sub-functions. The first two subfunctions 
are the evaluation of the function phi and its derivative 
according to the optimal sigma. 

The equation_p1 function calculates the optimal step using 
the Sherman Morrison Woodbury formula.  

The equation_p2 function calculates the optimal step using 
the Moore Penrose pseudoinverse matrix.

The equation_p3 function computes the optimal step using 
the Moore-Penrose pseudoinverse matrix and the unit 
vector from the eigenspace from the minimum eigenvalue 
from the extended eigenvalue vector.

The last function newton_method searches for the zero point
of the function phi using Newton's method and returns the
zero point as sigma.

The main part of this function represents the orthonormal 
basis solver, whose implementation can be looked up 
in the source mentioned above.

### Cauchy Point Calculation Solver Function
This function solves a trust region subproblem with 
the Cauchy Point Calculation. We have taken the 
implementation from the following source.

Jorge, Nocedal, & Stephen, J., Wright. (2006). Numerical Optimization. 
Springer Science+Business Media, LLC, 71-73
Available from 

https://www.csie.ntu.edu.tw/~r97002/temp/num_optimization.pdf

### Steihaug Conjugated Gradient Solver Function
This function solves a trust region subproblem with 
the Steihaugs Conjugated Gradient method. We have taken the 
implementation from the following source.

Jorge, Nocedal, & Stephen, J., Wright. (2006). Numerical Optimization. 
Springer Science+Business Media, LLC, 165-172

Available from https://www.csie.ntu.edu.tw/~r97002/temp/num_optimization.pdf.

We have taken some settings from the following source https://d-nb.info/1219852988/34.

Furthermore, we determined the steps by finding a tau 
by solving a quadratic equation.

### Step Function
The step function is the main function where the 
whole LSR1-TR algorithm is implemented. We would like 
to refer to the comments in the Python code and will 
briefly discuss only some details.

The gamma can be determined by two possibilities:  
Oleg, Burdakov, & Yu-Hong, Dai, & Na, Huang. (2019). Stabilizied Barzilai-Borwein
Method Available from https://arxiv.org/abs/1907.06409.
Finally, we determine the gamma as the maximum of the two variants at 0.1, i.e.
```
gamma = max(0.1, max(g_1, g_2))
```

The learning rate can generate NaN values in the loss 
through the Line Search. This happens when the resulting 
learning rate becomes very large or too close to zero. 
We catch this very unconventionally. We do similar 
checks with the loss and the gradient.
```
if 1e-12 > alpha_t or alpha_t > 1000000:
    state['restart'] = 1
    break
check_grad = torch.linalg.norm(flat_grad_t)
if check_grad < 1e-12 or check_grad > 1000000:
    state['restart'] = 1
    break
if loss_t > 1000000 or math.isnan(loss_t):
    state['restart'] = 1
    break
```
We are grateful for any improvement or solution 
to this problem that is communicated to us.

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

## Loss Function
We chose the cross-entropy loss function because it 
is very suitable for classifying objects into a fixed 
number of groups. 
It is also twice continuously differentiable, which 
is very important for our algorithm. 
A list of the possible loss functions offered 
by torch can be found at the following link.

https://pytorch.org/docs/stable/nn.html#loss-functions

It is essential that a second derivative exists, since 
our algorithm approximates the search direction using 
the second derivative.

## Hyper-parameters
* Learning rate: lr
* Max Iteration of the LSR1-TR Algorithm per Epoch: max_iter
* Break condition for the gradient: tolerance_grad
* Break condition of the actual reduction: tolerance_change
* Initial trust region radius: tr_radius
* Number of Columns of the LSR1 Matrix: memory_size
* Momentum term: mu
* Momentum term: nu
* Momentum term: alpha_S
* Max iteration of the Newton method: newton_maxit
* Max Iteration of the Steihaug Conjugated Gradient Method: cg_iter
* Option for a line search: line_search_fn
* The type of the trust region solver: trust_solver

The corresponding default values can be looked up in the function body.
We would like to discuss the hyperparameters briefly, 
some hyperparameters must be shared or some 
cannot be used when others are used.

### lr and line_search_fn
This implementation offers a line search. 
Alternatively, a fixed learning rate can be selected. 
If a fixed learning rate is selected, then 
```
line_search_fn = None 
```
must be set. Default value is line search with 
strong wolfe conditions.

### mu, nu and alpha_S
mu, nu and alpha_S are learning rates for momentum terms.
Where mu is the learning rate for a momentum term, 
nu is the learning rate for the s and alpha_S is the 
learning rate for the Gardient. The final search 
direction is then a linear combination of the three 
momentum terms or gradients and the search direction 
from the trust region subproblem.

It is recommended to set alpha_S =0 and choose 
nu and mu between 0 and 1.

### trust_solver, newton_maxit and cg_iter
There are three settings for trust_solver: 
"OBS", "Cauchy Point Calculation" and "Steihaug_cg". 
If "OBS" is selected then there is the option to 
select newton_maxit. This sets the maximum iteration 
of the Newton method for the orthonormal basis SR1 solver.
For the Steihaug_cg method, one can specify the maximum 
iteration by cg_iter.

### memory size
The memory size finally indicates how many columns the 
LSR1 matrix has. The larger the memory size the more 
computational effort. The best number has proven to be 12.
If the memory is not enough, it is advisable to reduce 
the memory size.

### tr_radius
This is where the initial radius for the Trust Region 
subproblem is set. 

### tolerance_change and tolerance_grad
These two hyperparameters are not very important. 
tolerance_change checks if the reduction of the 
target function is not too small. 
tolerance_grad checks if the norm of the gradient 
is not too small. In both cases a restart is initiated 
or the batch is changed.

## Potential problems
It may be that the system you are working on is 
somewhat older and cannot install or use the 
necessary dependencies of torch. 
This can be recognized by the following error message.
```
Segmentation fault (core dumped)
```
The solution is to install torch from source, 
for this you enter the following commands separately.
```
git clone --recursive https://github.com/pytorch/pytorch
pip install .
```
Here you should note that the versions of torch and 
torchvision are not the latest. 
This can lead to conflicts, also with other libraries. 
Further error handling is made more difficult here. 
In most cases the libraries have to be downgraded. 
One library that caused problems was Pillow, which 
we accordingly downgraded to 8.1.1 in this situation. 
However, this can also affect any other library individually.




