import torch
import torch.nn as nn
import torch.nn.functional as F



class Model(nn.Module):
    def __init__(self, input_size=1, output_size=1):
        super().__init__()

        # self.W = nn.Parameter(torch.zeros(1, 1))
        # self.b = nn.Parameter(torch.zeros(1))

        # Linear class
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        # y = x @ self.W + self.b
        y = self.linear(x)
        return y


def main():

    # toy dataset
    x = torch.rand(100, 1)
    y = 5 + 2 * x + torch.rand(100, 1)

    lr = 0.1
    iters = 100



    model = Model()

    # before training
    # get all parameters
    print('=== before training ===')
    for param in model.parameters(): 
        print(param)
    print('=======================')


    # generate optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    # taraining
    for i in range(iters):
        y_hat = model(x)
        loss = F.mse_loss(y, y_hat)

        loss.backward()
        
        # update parametes
        optimizer.step()

        # reset gradient
        optimizer.zero_grad()

    # after training
    # get all parameters
    print('=== after training ===')
    for param in model.parameters(): 
        print(param)
    print('======================')










if __name__ == '__main__':
    main()
