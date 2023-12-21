## 代码使用简介

1. 关联数据集，代码中datasets路径下是两套不同分类依据的蛋白质晶体数据集
2. 在`train.py`脚本中将`--data-path`设置成训练集文件夹绝对路径
3. 下载预训练权重，在`model.py`文件中每个模型都有提供预训练权重的下载地址，根据自己使用的模型下载对应预训练权重
4. 在`train.py`脚本中将`--weights`参数设成下载好的预训练权重路径，在weights分支中有本文所采用的权重
5. 设置好数据集的路径`--data-path`以及预训练权重的路径`--weights`就能使用`train.py`脚本开始训练了(训练过程中会自动生成`class_indices.json`文件)
6. 在`predict0.py`脚本中导入和训练脚本中同样的模型，并将`model_weight_path`设置成训练好的模型权重路径(默认保存在weights文件夹下)
7. 在`predict0.py`脚本中将`img_path`设置成你自己需要预测的图片绝对路径
8. 设置好权重路径`model_weight_path`和预测的图片路径`img_path`就能使用`predict0.py`脚本进行预测了
9. 在`predict_estimate.py`脚本中导入相同模型，设置测试集路径，进行模型评估
10. 如果要使用自己的数据集，请按照晶体分类数据集的文件结构进行摆放(即一个类别对应一个文件夹)，并且将训练以及预测脚本中的`num_classes`设置成你自己数据的类别数

所有数据均为实验室自采，使用的蛋白质晶体图片均为溶菌酶结晶所得。
原始数据集包括蛋白质晶体X射线衍射实验的有关数据放在kaggle上
download url: https://www.kaggle.com/datasets/superredworld/crystals-diffraction
