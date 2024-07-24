import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt




class Model(nn.Module):
    def __init__(self, input_size=1, hidden_size=10, output_size=1):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        y = self.linear1(x)
        y = F.sigmoid(y)
        y = self.linear2(y)
        return y


def main():
    torch.manual_seed(0)
    x = torch.rand(100, 1)
    y = torch.sin(2 * torch.pi * x) + torch.rand(100, 1)

    # plotting
    plt.scatter(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('generated_data.png')



    lr = 0.2
    iters = 10000

    model = Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)


    for i in range(iters):
        y_pred = model(x)
        loss = F.mse_loss(y, y_pred)

        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()

        if i % 1000 == 0:
            print(loss.item())
    print('==============')
    print('loss: ', loss.item())
    print('==============')


    # plotting
    plt.scatter(x, y)

    x = torch.arange(0, 1, 0.01)
    x = torch.unsqueeze(x, 1)
    y_pred = model(x)
    x = x.numpy()
    y_pred = y_pred.detach().numpy()

    plt.plot(x, y_pred)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('fitting.png')






if __name__ == '__main__':
    main()
