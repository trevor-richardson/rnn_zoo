# rnn_zoo
Test novel and important recurrent neural network architectures on baseline datasets SeqMNIST and pMNIST.

Architectures tested include:
* RNN
* LSTM - http://www.bioinf.jku.at/publications/older/2604.pdf
* GRU - https://arxiv.org/pdf/1406.1078v3.pdf
* IRNN - https://arxiv.org/abs/1504.00941
* Peephole LSTM - https://ieeexplore.ieee.org/document/861302/
* UGRNN - https://arxiv.org/pdf/1611.09913.pdf
* Intersection RNN - https://arxiv.org/pdf/1611.09913.pdf
* IndRNN - https://arxiv.org/pdf/1803.04831.pdf


## Results
The following results are were generated using the architectures listed above. \
Hyperparameters used: layers 3, num neurons 50, optimizer Adam, learning rate .0001 and batch 64.


<img src="https://github.com/trevor-richardson/rnn_zoo/blob/master/results/psmnist_val_acc.png" width="500">

<img src="https://github.com/trevor-richardson/rnn_zoo/blob/master/results/psmnist_train_acc.png" width="500">

<img src="https://github.com/trevor-richardson/rnn_zoo/blob/master/results/psmnist_val_loss.png" width="500">

<img src="https://github.com/trevor-richardson/rnn_zoo/blob/master/results/psmnist_train_loss.png" width="500">

<img src="https://github.com/trevor-richardson/rnn_zoo/blob/master/results/psmnist_train_acc.png" width="500">

<img src="https://github.com/trevor-richardson/rnn_zoo/blob/master/results/psmnist_train_acc.png" width="500">

<img src="https://github.com/trevor-richardson/rnn_zoo/blob/master/results/psmnist_train_acc.png" width="500">

<img src="https://github.com/trevor-richardson/rnn_zoo/blob/master/results/psmnist_train_acc.png" width="500">

## Running the code

```
  python train.py --model-type=irnn --task=seqmnist --layers=2 --batch-size=64 --epochs=10
```

### Installing

Update filepath in config.ini to where you've downloaded the repository.
Packages needed to run the code.
* numpy
* python3
* PyToch
* argparse
* configparser
