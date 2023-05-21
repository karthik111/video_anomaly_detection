import torch
import cifar_classifier
from PIL import Image
import torchvision.transforms.functional as TF
def main():
    PATH = './data/cifar10/cifar_net.pth'

    net = cifar_classifier.Net()
    net.load_state_dict(torch.load(PATH))

    image = Image.open(r'C:\\Users\karthik.venkat\PycharmProjects\video_anomaly_detection\data\cifar10\mafia.jpg')

    image = TF.resize(image, (32,32))
    image = TF.to_tensor(image)
    images = torch.unsqueeze(image,0)

    output = net(images)

    _, predicted = torch.max(output, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[0]]:5s}'))


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if __name__ == '__main__':
    main()