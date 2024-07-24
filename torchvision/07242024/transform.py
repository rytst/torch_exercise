import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt



def main():
    transform = transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(
        root='data',
        train=True,
        transform=transform,
        download=True,
    )

    print(dataset)
    x, label = dataset[0]

    print('type: ', type(x))
    print('shape: ', x.shape)

    print(x[0])
    plt.imshow(x[0])
    plt.savefig('transformed_mnist.png')



if __name__ == '__main__':
    main()
