import numpy as np
from homework1 import Hw1Env

env = Hw1Env(render_mode="gui")
for _ in range(100):
    print(_)
    env.reset()
    action_id = np.random.randint(4)
    _, img_before = env.state()
    env.step(action_id)
    pos_after, img_after = env.state()
    env.reset()
