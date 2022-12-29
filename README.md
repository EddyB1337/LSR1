# LSR1-TR Method for Deep Learning

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup with Google Colab](#setup-with-google-colab)
* [Setup via Terminal](#setup-via-terminal)
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


## Functions from torch

## Functions from me

## Deep Learning Model (CNN)

## Hyper-parameters

## Sources



