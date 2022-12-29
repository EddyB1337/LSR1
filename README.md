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

### Wolfe Conditions


## Functions from me

## Deep Learning Model (CNN)

## Hyper-parameters

## Sources



