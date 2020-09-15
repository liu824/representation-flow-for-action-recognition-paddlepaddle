# representation-flow-for-action-recognition-paddlepaddle
paper:https://arxiv.org/abs/1810.01455
#### 使用飞桨复现了representation flow for action recognition
### run model 
#### 训练模型
python train_model.py 
      -phase 'train'
      -batch_size 16
      -size 224
      -pretrain 'your pretrain model'
#### 测试模型
python train_model.py 
      -phase 'evel'
      -batch_size 16
      -size 224
      -model 'your model'
可以在下面的链接中找到我的项目 https://aistudio.baidu.com/aistudio/projectdetail/866797?shared=1
