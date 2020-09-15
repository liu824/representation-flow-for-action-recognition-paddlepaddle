# representation-flow-for-action-recognition-paddlepaddle
paper:https://arxiv.org/abs/1810.01455
使用飞桨复现了representation flow for action recognition
### run model 
#### if you want to train the model
run 
python train_model.py 
      -phase 'train'
      -batch_size 16
      -size 224
      -pretrain 'your pretrain model'
#### if you want to test the model  
python train_model.py 
      -phase 'evel'
      -batch_size 16
      -size 224
      -model 'your model'
you can get my project in https://aistudio.baidu.com/aistudio/projectdetail/866797?shared=1
