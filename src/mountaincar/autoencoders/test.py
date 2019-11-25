import torch
from autoencoder import Autoencoder
import numpy as np
from config import args
from dummyStates import flippedStates,shapeStates

# configure CUDA availability
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

autoencoder1=Autoencoder(args.input_shape_1,args.latent_shape)
autoencoder2=Autoencoder(args.input_shape_2,args.latent_shape)

if not use_cuda:
    checkpoint1 = torch.load(args.weight_paths[0],map_location='cpu')
    checkpoint2 = torch.load(args.weight_paths[1],map_location='cpu')
else:
    checkpoint1 = torch.load(args.weight_paths[0])
    checkpoint2 = torch.load(args.weight_paths[1])

autoencoder1.load_state_dict(checkpoint1['model_state_dict'])
autoencoder2.load_state_dict(checkpoint2['model_state_dict'])

autoencoder1.eval()
autoencoder2.eval()

if args.env == 'flipped':
    env = flippedStates(args.input_shape_1)
    original_states, new_states = env.getStates()
    
    k = np.random.randint(0,args.input_shape_1,1)
    state = original_states[k]
    new_state = new_states[k]

    state = (torch.FloatTensor(state)).unsqueeze(0)
    new_state = (torch.FloatTensor(new_state)).unsqueeze(0)

    s1,z1 = autoencoder1(state)
    s2,z2 = autoencoder2(new_state)

    print(s1,state)
    print('\n\n')
    print(s2,new_state)
    print('\n\n')
    print(z1,z2)

if args.env == 'shape':
    env = shapeStates(args.input_shape_1,args.input_shape_2)
    original_states, new_states = env.getStates(flip=args.flip,random=args.random)
    
    k = np.random.randint(0,args.input_shape_1,1)
    state = original_states[k]
    new_state = new_states[k]

    state = (torch.FloatTensor(state)).unsqueeze(0)
    new_state = (torch.FloatTensor(new_state)).unsqueeze(0)

    s1,z1 = autoencoder1(state)
    s2,z2 = autoencoder2(new_state)

    print(s1)
    print(state)
    print('\n\n')
    print(s2)
    print(new_state)
    print('\n\n')
    print(z1)
    print(z2)

