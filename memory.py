from pkgs import *


class PrioritizedReplay(object):
    def __init__(self, max_size, alpha=0.6, beta=0.4, beta_incr=1e-4):
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta
        self.beta_incr = beta_incr
        self.eps = 1e-5
        self.i = 1
        self.n = 0
        self.probs = np.zeros(2*max_size+10, dtype=np.float32)
        self.buffer = np.empty(2*max_size+10, dtype=object)

    def update_beta(self):
        self.beta = np.min([1., self.beta + self.beta_incr])

    def update_point(self, idx, td_error):
        self.probs[idx] = (np.abs(td_error)+self.eps)**self.alpha
        while idx > 1:
            p = idx//2
            self.probs[p] = self.probs[2*p]+self.probs[2*p+1]
            idx = p

    def sample_point(self, cur, wt):

        while cur <= self.max_size:
            left, right = 2*cur, 2*cur+1
            if self.probs[left] >= wt:
                cur = left
            else:
                wt -= self.probs[left]
                cur = right

        return cur

    def save(self, td_error, data):
        # data (s, a, r, s_, dones)
        if self.i > self.max_size:
            self.i = 1
        cur = self.i+self.max_size
        self.probs[cur] = (np.abs(td_error)+self.eps)**self.alpha
        self.buffer[cur] = data
        self.i += 1
        while cur > 1:
            p = cur//2
            self.probs[p] = self.probs[2*p]+self.probs[2*p+1]
            cur = p

        if self.n < self.max_size:
            self.n += 1

    def sample(self, batch_size):
        segment_size = self.probs[1]/batch_size
        batches, idxs, probs = [], [], []
        for i in range(batch_size):
            idx = self.sample_point(1, random.uniform(
                i*segment_size, (i+1)*segment_size))
            idxs.append(idx)
            batches.append(self.buffer[idx])
            probs.append(self.probs[idx])
        probs /= self.probs[1]
        IS_wts = (probs*self.n)**-self.beta
        IS_wts /= IS_wts.max()
        return batches, IS_wts, idxs
