import os
import json

import torch
from PIL import Image
from time import sleep
from tqdm import tqdm
import datetime
import numpy as np  # pip install numpy
import pandas as pd  # pip install pandas
from numpy import interp
from torchvision import transforms
from model import convnext_tiny as create_model
from sklearn.metrics import *  # pip install scikit-learn
import matplotlib.pyplot as plt  # pip install matplotlib
from sklearn.preprocessing import label_binarize

true_img = []
false_img = []
label_true = []
label_pre = []

label_list = []
likelihood_list = []
pred_list = []


def predict(img_path, true_cla, weight_path, num_class, size):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(f"using {device} device.")

    num_classes = num_class
    img_size = size
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
    model_weight_path = weight_path
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

        pred_softmax = torch.softmax(output, dim=0).cpu().numpy()
        # 获取可能性最大的标签
        label = torch.softmax(output, dim=0).cpu().numpy().argmax()
        label_list.append(label)
        # 获取可能性最大的值（即概率）
        likelihood = torch.softmax(output, dim=0).cpu().numpy().max()
        likelihood_list.append(likelihood)
        pred_list.append(pred_softmax.tolist())

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
    from model import convnext_tiny as create_model  # 修改使用模型

    '''指标日志'''
    log = open("record_distance.txt", mode="a", encoding="utf-8")  # 修改日志名称
    # img_dir = 'datasets/distance/classified_all'  # 修改测试集路径
    img_dir = 'datasets/distance/figure_enhanced_val'  # 修改测试集路径
    weight_path = "weights/distance/best_model(tiny).pth"  # 修改权重路径
    confusion_savepath = 'index/distance/confusion_all.png'
    num_class = 3  # 修改数量
    size = 224
    for i in os.listdir(img_dir):
        img_dir_new = img_dir + '/' + i
        img_name = os.listdir(img_dir_new)
        for j in tqdm(range(len(img_name))):
            path = os.path.join(img_dir_new, img_name[j])
            true_class = os.path.split(img_dir_new)[1][-1]
            predict(path, true_class, weight_path, num_class, size)
            sleep(0.01)

    # 获取当前时间
    current_time = datetime.datetime.now()
    # 将时间格式化为字符串
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    # 打印当前时间
    print("当前时间：", formatted_time, file=log)
    # 打印当前数据集
    print('当前数据集：', img_dir, file=log)
    # 打印当前模型
    print('当前模型：', weight_path, file=log)
    # 打印真实标签
    print(label_true, file=log)
    # 打印预测标签
    print(label_pre, file=log)
    # 打印真实标签个数
    print(len(true_img), file=log)
    # 打印预测标签个数
    print(len(false_img), file=log)
    # 打印分类报告(precision  recall  f1-score)
    print(classification_report(label_pre, label_true, target_names=['level1', 'level2', 'level3']), file=log)
    # 打印混淆矩阵
    C = confusion_matrix(label_true, label_pre, labels=['1', '2', '3'], normalize='true')  # 可将'1'等替换成自己的类别，如'cat'。
    print(C, file=log)

    log.close()

    label_names = ["1", "2", "3"]  # 可以把标签写在这里
    df_pred = pd.DataFrame(data=pred_list, columns=label_names)

    df_pred.to_csv('pred_result.csv', encoding='gbk', index=False)
    print("Done!")

    '''混淆矩阵'''
    plt.matshow(C, cmap=plt.cm.Reds)
    # plt.colorbar()

    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(round(C[j, i], 4), xy=(i, j), horizontalalignment='center', verticalalignment='center')

    plt.tick_params(labelsize=10)  # 设置左边和上面的label类别如0,1,2,3,4的字体大小。

    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    plt.ylabel('True label', fontdict={'family': 'Times New Roman', 'size': 20})  # 设置字体大小。
    plt.xlabel('Predicted label', fontdict={'family': 'Times New Roman', 'size': 20})
    plt.xticks(range(0, 3), labels=['level1', 'level2', 'level3'])  # 将x轴或y轴坐标，刻度 替换为文字/字符
    plt.yticks(range(0, 3), labels=['level1', 'level2', 'level3'], rotation=90)
    plt.show()
    plt.savefig(confusion_savepath)

    predict_loc = "pred_result.csv"
    predict_data = pd.read_csv(predict_loc)  # ,index_col=0)
    predict_label = predict_data.to_numpy().argmax(axis=1)
    predict_score = predict_data.to_numpy().max(axis=1)

    # '''
    #     常用指标：精度，查准率，召回率，F1-Score
    # '''
    # # 精度，准确率， 预测正确的占所有样本种的比例
    # accuracy = accuracy_score(label_true, predict_label)
    # print("精度: ", accuracy)
    #
    # # 查准率P（准确率），precision(查准率)=TP/(TP+FP)
    #
    # precision = precision_score(label_true, predict_label, labels=None, pos_label=1,
    #                             average='macro')  # 'micro', 'macro', 'weighted'
    # print("查准率P: ", precision)
    #
    # # 查全率R（召回率），原本为对的，预测正确的比例；recall(查全率)=TP/(TP+FN)
    # recall = recall_score(label_true, predict_label, average='macro')  # 'micro', 'macro', 'weighted'
    # print("召回率: ", recall)
    #
    # # F1-Score
    # f1 = f1_score(label_true, predict_label, average='macro')  # 'micro', 'macro', 'weighted'
    # print("F1 Score: ", f1)

    '''
    ROC曲线（多分类）
    在多分类的ROC曲线中，会把目标类别看作是正例，而非目标类别的其他所有类别看作是负例，从而造成负例数量过多，
    虽然模型准确率低，但由于在ROC曲线中拥有过多的TN，因此AUC比想象中要大
    '''
    n_classes = len(label_names)
    # binarize_predict = label_binarize(predict_label, classes=[i for i in range(n_classes)])
    binarize_predict = label_binarize(label_true, classes=['1', '2', '3'])

    # 读取预测结果
    #
    predict_score = predict_data.to_numpy()

    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(binarize_predict[:, i], [socre_i[i] for socre_i in predict_score])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # print("roc_auc = ",roc_auc)

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.4f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=lw, label='ROC curve of {0} (area = {1:0.4f})'.format(label_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Classification Characteristics of Deep Learning Networks ')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig('index/distance/ROC_all.png')
