# Transfer-Matching-Networks

#### Go to this [link](https://github.com/nsidn98/Transfer-Matching-Networks/blob/master/src/algos/dqn/README.md) for usage of zero-shot learning of cartpole-env


## Flipped States:
States is flipped in new environment:
* To run the basic experiment with flipped states:
`python autoencoder.py`

* To test the states, reconstruction, latent representation:
`python test.py`

## Different Shape States:
State for new environment is [0,state,0] (0 appended at start and end)
* To train: `python autoencoder.py --tensorboard=0 --input_shape_2=22 --env=shape`
* To test the states, reconstruction, latent representation:`python3 test.py --input_shape_2=22 --env=shape`

### Note: Please make sure to change the `weights_path` in `config.py`

## Requirements:
* torch
* tensorboardx
* tqdm
