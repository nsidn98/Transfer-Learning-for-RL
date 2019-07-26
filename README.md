# Transfer-Matching-Networks
* To run the basic experiment with flipped states:

`python autoencoder.py`

* To run with added gaussian noise with zero mean and 0.1 std:

`python autoencoder.py --noise=1 --noise_mean=0 --noise_std=0.1`

## Requirements:
* torch
* tensorboardx
* tqdm
