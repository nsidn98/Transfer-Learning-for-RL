from mountaincar import *

envorig = MountainCarEnv()
envtarget = MountainCarEnv(color=[0.5,0.5,0.5,0.5])
np_r, seed = seeding.np_random(None)
init_state = np.array([np_r.uniform(low=-0.6, high=-0.4), 0])
envorig.reset(init_state)
envtarget.reset(init_state)

for i in range(100000):
    
    action = np.random.randint(0, envorig.action_space.n)
    _,_,_,_ = envorig.step(action)
    _,_,_,_ = envtarget.step(action)
    envorig.render()
    envtarget.render()
env.close()
