# rnn_zoo
Test novel and important rnn architectures on baseline dataset SeqMNIST and pMNIST

Architectures tested include:
* RNN
* LSTM - http://www.bioinf.jku.at/publications/older/2604.pdf
* GRU - https://arxiv.org/pdf/1406.1078v3.pdf
* IRNN - https://arxiv.org/abs/1504.00941
* Peephole LSTM - ftp://ftp.idsia.ch/pub/juergen/TimeCount-IJCNN2000.pdf
* UGRNN - https://arxiv.org/pdf/1611.09913.pdf
* Intersection RNN - https://arxiv.org/pdf/1611.09913.pdf
* IndRNN - https://arxiv.org/pdf/1803.04831.pdf


## Results
The following results are were generated using the architectures listed above.
Hyperparameters used: layers 3, num neurons 50, optimizer Adam, learning rate .0001 and batch 64.


<img src="https://github.com/trevor-richardson/rnn_zoo/blob/master/results/psmnist_val_acc.png" width="500">

<img src="https://github.com/trevor-richardson/rnn_zoo/blob/master/results/psmnist_train_acc.png" width="500">

<img src="https://github.com/trevor-richardson/rnn_zoo/blob/master/results/psmnist_val_loss.png" width="500">

<img src="https://github.com/trevor-richardson/rnn_zoo/blob/master/results/psmnist_train_loss.png" width="500">
\

---

\

<img src="https://github.com/trevor-richardson/rnn_zoo/blob/master/results/psmnist_train_acc.png" width="500">

<img src="https://github.com/trevor-richardson/rnn_zoo/blob/master/results/psmnist_train_acc.png" width="500">

<img src="https://github.com/trevor-richardson/rnn_zoo/blob/master/results/psmnist_train_acc.png" width="500">

<img src="https://github.com/trevor-richardson/rnn_zoo/blob/master/results/psmnist_train_acc.png" width="500">

## Running the code

### Installing

Update filepath in config.ini to where you've downloaded the repository.
Packages needed to run the code.
* numpy
* scipy
* python3
* pytorch
* matplotlib
* pygame
* pylab
* glob
