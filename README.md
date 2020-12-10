# Log-Linear Model for Spam Classification

## Usage

### positional & optional args

- mode : choose 'train' to train on enron 1-5, 'test' to test on enron 6 and 'optimize' to find optimal parameters
- dir_path : path to the dir containing the enron datasets
- load_path : path to the JSON file holding the weights
  
- --lr : learning rate. default: 0.001
- --l2 : regularization factor. default: 0.01
- --epochs : nr of iterations over the dataset. default: 4
  
### testing

Test on enron 6. Best observed test accuracy: 97,3% (on default args)

```
python3 log-linear.py test dir_path=path_to_datadir load_path=path_to_weights
```

### training

train on enron 1-5. Best observed development accuracy: 98,1% (on default args)

```
python3 log-linear.py train dir_path=path_to_datadir load_path=path_to_weights --lr=0.001 --l2=0.01 --epochs=5
```

### optimization

will train over several hyperparameter settings to fin the optimal one.

```
python3 log-linear optimize dir_path=path_to_datadir load_path=path_to_weights
```
