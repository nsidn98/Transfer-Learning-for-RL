# DisentAngled Representation LearnING(DARLING)
## Usage:(follow sequentially)
* First run `python3 dqn.py --n_episode=500 --max-episode=100` to train the autoencoders on random cartpole states
* Then run `python3 darlingDQN.py --n_episodes=1000` to train the DQN with states as the latent representation(size=8) of the autoencoders with the original states as the inputs to the autoencoder.
* Then to test with original states run `python3 darlingDQN.py --n_episode=100 --test=1 --state_type=original`
* To test with flipped states as input to the autoencoder `python3 darlingDQN.py --n_episode=100 --test=1 --state_type=target`
