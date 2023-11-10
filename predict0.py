import os
import json

import torch
import shutil
from PIL import Image
from time import sleep
from tqdm import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt

from model import convnext_tiny as create_model


def predict(img_path):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(f"using {device} device.")

    num_classes = 3
    img_size = 224
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    img_path = "data/bohe/000119_undist_39_1.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    # plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = create_model(num_classes=num_classes).to(device)
    # load model weights
    model_weight_path = "weights/best_model(eca_kernel5 in stage).pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    a = max(predict.numpy())
    # print(a,class_indict[str(predict_cla)])
    img_name = os.path.basename(img_path)
    # print(img_name)
    if a > 0.85:
        shutil.copy(img_path,'data/predict_top/'+class_indict[str(predict_cla)]+'/'+str(img_name))
    else:
        shutil.copy(img_path, 'data/predict_top/其他/' + str(img_name))
    # plt.show()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")
    img_dir = 'data/top'
    img_name = os.listdir(img_dir)
    for i in tqdm(range(len(img_name))):
        path = os.path.join(img_dir, img_name[i])
        predict(path)
        sleep(0.01)
