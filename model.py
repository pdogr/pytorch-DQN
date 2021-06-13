from pkgs import *


class CartPoleDNN(nn.Module):
    def __init__(self, input_dims, action_dim):
        super(CartPoleDNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dims[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)


class CartPoleDuelDNN(nn.Module):
    def __init__(self, input_dims, action_dim):
        super(CartPoleDuelDNN, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_dims[0], 128),
            nn.ReLU()
        )
        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.adv = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        x = self.fc1(x)
        V = self.adv(x)
        A = self.adv(x)
        return V+(A-A.mean())
