import torch



def rosenbrock(x0, x1):
    y = 100 * (x1 - x0**2)**2 + (x0 - 1)**2
    return y



def gradient_descent(x0, x1, lr=1e-3, N=10000):
    for n in range(N):
        y = rosenbrock(x0, x1)
        y.backward()
        x0.data -= lr * x0.grad.data
        x1.data -= lr * x1.grad.data


        # reset grad
        x0.grad.zero_()
        x1.grad.zero_()
    return (x0, x1)
    



def main():
    x0 = torch.tensor(0., requires_grad=True)
    x1 = torch.tensor(2., requires_grad=True)


    y = rosenbrock(x0, x1)
    y.backward()
    print('grad: (', x0.grad, ',', x1.grad, ')' )

    x0 = torch.tensor(0., requires_grad=True)
    x1 = torch.tensor(2., requires_grad=True)

    x0, x1 = gradient_descent(x0, x1, lr=1e-3, N=10000)
    print(x0, x1)



if __name__ == '__main__':
    main()
