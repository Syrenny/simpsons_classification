import torch
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt


def unnormalize(image):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    unnorm = transforms.Normalize((-1 * mean[0] / std[0], -1 * mean[1] / std[1],
                                        -1 * mean[2] / std[2]), (1.0 / std[0], 1.0 / std[1], 1.0 / std[2]))
    return unnorm(image)


def final_testing(path, name_classes, loader, device):
    model = torch.load(path)
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader['test']):
            inputs = inputs.to(device)
            outputs = model(inputs)
            to_probability = nn.Softmax(dim=1)
            outputs = to_probability(outputs)
            _, preds = torch.max(outputs, 1)
            for j, idx in enumerate(preds):
                probability = outputs[j][idx].cpu().numpy()
                image = unnormalize(inputs[j]).permute(1, 2, 0).cpu().numpy()
                predicted_label = name_classes[idx]  
                real_label = name_classes[labels[j].numpy()]
                print("Predicted: '{}' with probability {:.2f}. Expected: '{}'\n".format(predicted_label, probability, real_label))
                plt.imshow(image)
                plt.show()


