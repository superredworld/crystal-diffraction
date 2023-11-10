import os
import json

import torch
from PIL import Image
from time import sleep
from tqdm import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt
from model import convnext_tiny as create_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

true_img = []
false_img = []
label_true = []
label_pre = []

def predict(img_path, true_cla):
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
    # img_path = "valid_sets/crystal_level3/174.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    # plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # plt.imshow(img)
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
    model_weight_path = "weights/best_model.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    label_true.append(true_cla)
    label_pre.append(class_indict[str(predict_cla)][13])

    # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
    #                                              predict[predict_cla].numpy())
    # plt.title(print_res)
    # for i in range(len(predict)):
    #     print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
    #                                               predict[i].numpy()))
    if class_indict[str(predict_cla)][13] == true_cla:
        true_img.append(img_path)
    else:
        false_img.append(img_path)

    # plt.show()



if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    log = open("record.txt", mode="a", encoding="utf-8")
    img_dir = 'classified_all'
    for i in os.listdir(img_dir):
        img_dir_new = img_dir+'/'+i
        img_name = os.listdir(img_dir_new)
        for j in tqdm(range(len(img_name))):
            path = os.path.join(img_dir_new, img_name[j])
            true_class = os.path.split(img_dir_new)[1][-1]
            predict(path, true_class)
            sleep(0.01)
    # 预测标签
    print(label_pre, file=log)
    # 真实标签
    print(label_true, file=log)
    # 正确标签个数
    print(len(true_img), file=log)
    # 错误标签个数
    print(len(false_img), file=log)
    # 打印分类报告（）
    print(classification_report(label_true, label_pre, target_names=['level1', 'level2', 'level3']), file=log)

    C = confusion_matrix(label_true, label_pre, labels=['1', '2', '3'])  # 可将'1'等替换成自己的类别，如'cat'。
    print(C, file=log)

    log.close()

    plt.matshow(C, cmap=plt.cm.Reds)
    # plt.colorbar()

    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')

    # plt.tick_params(labelsize=15) # 设置左边和上面的label类别如0,1,2,3,4的字体大小。

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.ylabel('True label', fontdict={'family': 'Times New Roman', 'size': 20}) # 设置字体大小。
    # plt.xlabel('Predicted label', fontdict={'family': 'Times New Roman', 'size': 20})
    # plt.xticks(range(0,5), labels=['a','b','c','d','e']) # 将x轴或y轴坐标，刻度 替换为文字/字符
    # plt.yticks(range(0,5), labels=['a','b','c','d','e'])
    plt.show()
