from pkgs import *
from memory import PrioritizedReplay


class DQNAgent(object):
    def __init__(self, initial_eps, min_eps, gamma, final_explore_frame,
                 actions, model, model_params,
                 C, batch_size, optimizer_params,
                 replay_size, replay_start,
                 device='cpu'):
        self.epsilon = initial_eps
        self.min_eps = min_eps
        self.gamma = gamma
        self.actions = actions
        self.batch_size = batch_size
        self.replay_start = replay_start
        self.memory = PrioritizedReplay(replay_size)
        self.device = device
        self.steps = 0
        self.C = C
        self.final_explore_frame = final_explore_frame
        self.initial_eps = initial_eps
        self.min_eps = min_eps

        self.q = model(**model_params).to(device)
        self.q.apply(self.weights_init)
        self.q_h = model(**model_params).to(device)
        self.optimizer = optim.RMSprop(
            params=self.q.parameters(), lr=optimizer_params.get('lr'))
        self.update_network()

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            T.nn.init.xavier_uniform_(m.weight)

    def action(self, obs):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            with T.no_grad():
                x = T.tensor([obs], dtype=T.float).to(self.device)
                out = self.q.forward(x)
                return T.argmax(out).item()

    def update_network(self):
        self.q_h.load_state_dict(self.q.state_dict())

    def update_epsilon(self):
        self.epsilon = self.min_eps + \
            (self.initial_eps-self.min_eps) * \
            math.exp(-1.*self.steps/self.final_explore_frame)

    def update_memory(self):
        self.memory.update_beta()

    def y_target(self, R, S_, dones):
        Q_next = self.q_h.forward(S_).max(dim=1)[0]
        return R+self.gamma*Q_next*(1-dones)

    def save(self, s, a, r, s_, done):
        with T.no_grad():
            Q = self.q.forward(T.FloatTensor([s]).to(self.device))
            Q_next = self.q_h.forward(T.FloatTensor([s_]).to(self.device))
            Y_target = r + (1-done)*self.gamma*T.max(Q_next)
            td_error = (Y_target - Q[0][a]).cpu()
            self.memory.save(td_error, (s, a, r, s_, done))

    def sample(self):
        data, IS_wts, idxs = self.memory.sample(self.batch_size)
        data = np.array(data, dtype=object).transpose()
        (S, A, R, S_, dones) = (np.stack(data[0], axis=0), list(
            data[1]), list(data[2]), np.stack(data[3], axis=0), list(data[4]))
        S = T.tensor(S, dtype=T.float).to(self.device)
        R = T.tensor(R, dtype=T.float).to(self.device)
        S_ = T.tensor(S_, dtype=T.float).to(self.device)
        dones = T.tensor(dones, dtype=T.long).to(self.device)
        IS_wts = T.tensor(IS_wts, dtype=T.float).to(self.device)
        return (S, A, R, S_, dones), IS_wts, idxs

    def step(self):
        self.steps += 1
        if self.steps < self.replay_start:
            return np.inf

        if self.steps % self.C == 0:
            self.update_network()

        (S, A, R, S_, dones), IS_wts, idxs = self.sample()
        Q = self.q.forward(S)[np.arange(self.batch_size), A]
        Y_target = self.y_target(R, S_, dones)

        loss = (Y_target-Q)
        TD_errors = loss.cpu().data.numpy()
        for i, idx in enumerate(idxs):
            self.memory.update_point(idx, TD_errors[i])
        loss = IS_wts * loss.pow(2)
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_epsilon()
        self.update_memory()
        return loss.item()

    def save_model(self, filename):
        T.save({
            'q': self.q.state_dict(),
            'q_h': self.q_h.state_dict(),
            'epsilon': self.epsilon,
        }, filename)

    def load_model(self, filename):
        ckpt = T.load(filename)
        self.q.load_state_dict(ckpt['q'])
        self.q_h.load_state_dict(ckpt['q_h'])
        self.epsilon = ckpt['epsilon']


class DDQNAgent(object):
    def __init__(self, initial_eps, min_eps, gamma, final_explore_frame,
                 actions, model, model_params,
                 C, batch_size, optimizer_params,
                 replay_size, replay_start,
                 device='cpu'):
        self.epsilon = initial_eps
        self.min_eps = min_eps
        self.gamma = gamma
        self.actions = actions
        self.batch_size = batch_size
        self.replay_start = replay_start
        self.memory = PrioritizedReplay(replay_size)
        self.device = device
        self.steps = 0
        self.C = C
        self.final_explore_frame = final_explore_frame
        self.initial_eps = initial_eps
        self.min_eps = min_eps

        self.q = model(**model_params).to(device)
        self.q.apply(self.weights_init)
        self.q_h = model(**model_params).to(device)
        self.optimizer = optim.RMSprop(
            params=self.q.parameters(), lr=optimizer_params.get('lr'))
        self.update_network()

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            T.nn.init.xavier_uniform_(m.weight)

    def action(self, obs):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            with T.no_grad():
                x = T.tensor([obs], dtype=T.float).to(self.device)
                out = self.q.forward(x)
                return T.argmax(out).item()

    def update_network(self):
        self.q_h.load_state_dict(self.q.state_dict())

    def update_epsilon(self):
        self.epsilon = self.min_eps + \
            (self.initial_eps-self.min_eps) * \
            math.exp(-1.*self.steps/self.final_explore_frame)

    def update_memory(self):
        self.memory.update_beta()

    def y_target(self, R, S_, dones):
        with T.no_grad():
            Q = self.q.forward(S_)
            actions = T.argmax(Q, dim=1)
            Q_next = self.q_h.forward(S_)
            return R+self.gamma*Q_next[np.arange(len(dones)), actions]*(1-dones)

    def save(self, s, a, r, s_, done):
        with T.no_grad():
            Q = self.q.forward(T.FloatTensor([s]).to(self.device))
            Q_next = self.q_h.forward(T.FloatTensor([s_]).to(self.device))
            Y_target = r + (1-done)*self.gamma*T.max(Q_next)
            td_error = (Y_target - Q[0][a]).cpu()
            self.memory.save(td_error, (s, a, r, s_, done))

    def sample(self):
        data, IS_wts, idxs = self.memory.sample(self.batch_size)
        data = np.array(data, dtype=object).transpose()
        (S, A, R, S_, dones) = (np.stack(data[0], axis=0), list(
            data[1]), list(data[2]), np.stack(data[3], axis=0), list(data[4]))
        S = T.tensor(S, dtype=T.float).to(self.device)
        R = T.tensor(R, dtype=T.float).to(self.device)
        S_ = T.tensor(S_, dtype=T.float).to(self.device)
        dones = T.tensor(dones, dtype=T.long).to(self.device)
        IS_wts = T.tensor(IS_wts, dtype=T.float).to(self.device)
        return (S, A, R, S_, dones), IS_wts, idxs

    def step(self):
        self.steps += 1
        if self.steps < self.replay_start:
            return np.inf

        if self.steps % self.C == 0:
            self.update_network()

        (S, A, R, S_, dones), IS_wts, idxs = self.sample()
        Q = self.q.forward(S)[np.arange(self.batch_size), A]
        Y_target = self.y_target(R, S_, dones)

        loss = (Y_target-Q)
        TD_errors = loss.cpu().data.numpy()
        for i, idx in enumerate(idxs):
            self.memory.update_point(idx, TD_errors[i])
        loss = IS_wts * loss.pow(2)
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_epsilon()
        self.update_memory()
        return loss.item()

    def save_model(self, filename):
        T.save({
            'q': self.q.state_dict(),
            'q_h': self.q_h.state_dict(),
            'epsilon': self.epsilon,
        }, filename)

    def load_model(self, filename):
        ckpt = T.load(filename)
        self.q.load_state_dict(ckpt['q'])
        self.q_h.load_state_dict(ckpt['q_h'])
        self.epsilon = ckpt['epsilon']
