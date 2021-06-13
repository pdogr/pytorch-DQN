from pkgs import *


class SkipMaxFrameWrapper(gym.Wrapper):
    def __init__(self, action_repeat, no_ops, env=None):
        super(SkipMaxFrameWrapper, self).__init__(env)
        self.action_repeat = action_repeat
        self.buffer = [np.zeros(self.env.observation_space.low.shape)]*2
        self.i = 0
        self.no_ops = no_ops

    def reset(self):
        state = self.env.reset()
        self.buffer[0] = state
        self.i = 1

        for i in range(np.random.randint(self.no_ops)):

            obs, _, done, _ = self.env.step(0)
            self.buffer[self.i] = obs
            self.i ^= 1
            if done:
                self.env.reset()
        return np.maximum(*self.buffer)

    def step(self, action):
        r = 0.0
        done = False
        for _ in range(self.action_repeat):
            obs, r_i, done, info = self.env.step(action)
            r += r_i
            self.buffer[self.i] = obs
            self.i ^= 1
            if done:
                break
        return np.maximum(*self.buffer), r, done, info


class GrayResizeWrapper(gym.ObservationWrapper):
    def __init__(self, dims, env=None):
        super(GrayResizeWrapper, self).__init__(env)
        self.dims = dims
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                                                shape=self.dims, dtype=np.float32)

    def observation(self, obs):
        img = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        new_obs = cv2.resize(
            img, (self.dims[1], self.dims[2]), interpolation=cv2.INTER_LINEAR)

        return (np.array(new_obs, dtype=np.uint8).reshape(self.dims))/255.0


class StackWrapper(gym.ObservationWrapper):
    def __init__(self, m, env=None):
        super(StackWrapper, self).__init__(env)
        self._stack = deque(maxlen=m)
        self.m = m
        self.observation_space = gym.spaces.Box(env.observation_space.low.repeat(m, axis=0),
                                                env.observation_space.high.repeat(m, axis=0), dtype=np.float32)

    def observation(self, obs):
        self._stack.append(obs)
        return np.array(self._stack).reshape(self.observation_space.high.shape)

    def reset(self):
        obs = self.env.reset()
        self._stack = deque(maxlen=self.m)
        for _ in range(self.m):
            self._stack.append(obs)
        return np.array(self._stack).reshape(self.observation_space.high.shape)


def make_env(name, dims, action_repeat, history_length, no_ops):
    env = gym.make(name)
    env = SkipMaxFrameWrapper(action_repeat, no_ops=no_ops, env=env)
    env = GrayResizeWrapper(dims=dims, env=env)
    env = StackWrapper(m=history_length, env=env)
    return env
