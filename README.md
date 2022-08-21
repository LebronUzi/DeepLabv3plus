# DeepLabv3plus
## logs
存储训练好的模型
## pretrained
预训练好的deeeplabv3+
## weizmann_horse_db
数据集以及预测的结果
## 代码介绍以及使用方法
### 代码功能
* annotation.py用于分割数据集
* Mobilnetv2.py是backbone
* model.py是构造网络模型
* Mydataset.py制作数据集并进行数据预处理
* train.py用于训练网络
* predict.py生成结果
* iou.py对结果进行评估

### 使用方法
* 首先使用annotation脚本分割数据集，并生成包含数据集图片名字的文件
* 然后利用预训练好的模型，运行train文件生成训练好的模型，注意模型的输出路径
* 找到训练好的模型，运行predict脚本输出结果，可以输出一张图片看效果，也可以将整个数据集输出
* 最后用mIou和boundary Iou评估结果
