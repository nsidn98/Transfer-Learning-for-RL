from mountaincar import *

env_orig = MountainCarEnv()
env_target = MountainCarEnv(color=[1,0,0,0.5])
np_r, seed = seeding.np_random(None)
init_state = np.array([np_r.uniform(low=-0.6, high=-0.4), 0])
env_orig.reset(init_state)
env_target.reset(init_state)

for i in range(100000):
    
    action = np.random.randint(0, env_orig.action_space.n)
    s,_,_,_ = env_orig.step(action)
    _,_,_,_ = env_target.step(action)
    img = env_orig.render(mode='rgb_array')
    img2 = env_target.render(mode='rgb_array')
    # print(np.mean(img),np.mean(img2))
    print(img.shape)
env_orig.close()
env_target.close()
