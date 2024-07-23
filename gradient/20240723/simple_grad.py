import torch

def main():
    x = torch.tensor(10., requires_grad=True)
    y = 2 * x
    y.backward()
    print(x.grad)
    


if __name__ == '__main__':
    main()
       
