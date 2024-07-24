import torchvision
import matplotlib.pyplot as plt



def main():
    dataset = torchvision.datasets.MNIST(
        root='data',
        train=True,
        transform=None,
        download=True,
    )


    x, label = dataset[0]

    print('size', len(dataset))
    print('type', type(dataset))
    print('label', label)


    plt.imshow(x, cmap='gray')
    plt.savefig('mnist.png')




if __name__ == '__main__':
    main()
