# rnn_zoo
This repository tests various recurrent neural network architectures on baseline datasets SeqMNIST and pMNIST. <br/>
The network architectures chosen were those deemed to be the most effective currently available.

Architectures tested include:
* **RNN**

 &nbsp;&nbsp;<img src="https://github.com/trevor-richardson/rnn_zoo/blob/master/results/rnn.png" width="300" />

* **LSTM** - http://www.bioinf.jku.at/publications/older/2604.pdf

<img src="https://github.com/trevor-richardson/rnn_zoo/blob/master/results/lstm.png" width="440" />

* **GRU** - https://arxiv.org/pdf/1406.1078v3.pdf

&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/trevor-richardson/rnn_zoo/blob/master/results/gru.png" width="640" />

* **IRNN** - https://arxiv.org/abs/1504.00941

&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/trevor-richardson/rnn_zoo/blob/master/results/irnn.png" width="320" />

* **Peephole LSTM** - https://ieeexplore.ieee.org/document/861302/

&nbsp;&nbsp;<img src="https://github.com/trevor-richardson/rnn_zoo/blob/master/results/plstm.png" width="450" />

* **UGRNN** - https://arxiv.org/pdf/1611.09913.pdf

&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/trevor-richardson/rnn_zoo/blob/master/results/ugrnn.png" width="370" />

* **Intersection RNN** - https://arxiv.org/pdf/1611.09913.pdf

&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/trevor-richardson/rnn_zoo/blob/master/results/intersectionrnn.png" width="370" />


## Results
The following results are were generated using the architectures listed above. \
Hyperparameters used: layers 3, num neurons 50, optimizer Adam, learning rate .0001 and batch 64.

<p float="left">
  <img src="https://github.com/trevor-richardson/rnn_zoo/blob/master/results/smnist_val_acc.png" width="440" />
  <img src="https://github.com/trevor-richardson/rnn_zoo/blob/master/results/smnist_train_acc.png" width="440" />
</p>

---

<p float="left">
  <img src="https://github.com/trevor-richardson/rnn_zoo/blob/master/results/smnist_val_loss.png" width="440" />
  <img src="https://github.com/trevor-richardson/rnn_zoo/blob/master/results/smnist_train_loss.png" width="440" />
</p>

---

<p float="left">
  <img src="https://github.com/trevor-richardson/rnn_zoo/blob/master/results/psmnist_val_acc.png" width="440" />
  <img src="https://github.com/trevor-richardson/rnn_zoo/blob/master/results/psmnist_train_acc.png" width="440" />
</p>

---

<p float="left">
  <img src="https://github.com/trevor-richardson/rnn_zoo/blob/master/results/psmnist_val_loss.png" width="440" />
  <img src="https://github.com/trevor-richardson/rnn_zoo/blob/master/results/psmnist_train_loss.png" width="440" />
</p>


## Running the code

```
  python train.py --model-type=irnn --task=seqmnist --layers=2 --batch-size=64 --epochs=10
```

### Installing

Update BASE_DIR in config.ini with the absolute path to the current directory. <br/>
Packages needed to run the code include:

* numpy
* python
* PyToch
* argparse
* configparser
