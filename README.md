# minimalist-self-attention
As the name suggests the code in the `main` branch is written in a minimalist way with only one script/module. The directory structure looks like this:
```bash
.
└── src
    └── vit.py
├── LICENSE
├── README.md
├── requirements.txt
```
In order to run this, there are only really 2 steps:

1. Create a virtual environment and install all the libraries
```bash
$ conda create -n minimalist-self-attention python=3.10.9
$ pip install -r requirements.txt
```
2. Run the `src/vit` module from the `./src` diretory:
``` bash
$ python vit.py
```

It should first download the MNIST dataset and your directory structure will look something like this:
```bash
.
├── datasets
│   └── MNIST
│       └── raw
│           ├── t10k-images-idx3-ubyte
│           ├── t10k-images-idx3-ubyte.gz
│           ├── t10k-labels-idx1-ubyte
│           ├── t10k-labels-idx1-ubyte.gz
│           ├── train-images-idx3-ubyte
│           ├── train-images-idx3-ubyte.gz
│           ├── train-labels-idx1-ubyte
│           └── train-labels-idx1-ubyte.gz
├── LICENSE
├── README.md
├── requirements.txt
└── src
    └── vit.py
```
And then the training will begin and on the terminal it should be something like this:
```
$ python vit.py 
Using device:  cuda (NVIDIA GeForce GTX 1050)
Training:   0%|                                          0/5 [00:00<?, ?it/s]
Epoch 1 in training:   25%|██████▍             | 20/469 [00:45<16:43, 2.23s/it]
```

## TODO
Refactor the code in stages such that each branch is just one level up with the idea od making a visual transformer that has all the bells and whistle like logger, experiment tracker, unit tests and docker

## Contribution Guidelines
This project was made primarily so that I can understand the transformer mechanism better, however, if you fee like contributing, then just raise an issue and we'll discuss there.