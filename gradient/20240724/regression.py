import torch


def predict(x, W, b):
    y = x @ W + b
    return y


def mean_squared_error(x0, x1):
    diff = x0 - x1
    N = len(diff)
    return diff.T @ diff / N




def main():
    # toy dataset
    torch.manual_seed(0)
    x = torch.rand(100, 1)
    y = 5 + 2 * x + torch.rand(100, 1)
    
    W = torch.zeros(1, 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    

    lr = 0.1
    iters = 100


    for i in range(iters):
        y_hat = predict(x, W, b)
        loss = mean_squared_error(y, y_hat)


        loss.backward()

        W.data -= W.grad.data 
        b.data -= b.grad.data

        W.grad.zero_()
        b.grad.zero_()


        if i % 10 == 0:
            print(loss.item())



if __name__ == '__main__':
    main()
