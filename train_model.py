import os
import argparse
import numpy as np
import paddle.fluid as fluid
import time
import json 
import pickle
import reader
import resnet_2p1d_model as MODEL

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, default='flow',help='rgb or flow')
    parser.add_argument('-exp_name', type=str)
    parser.add_argument('-batch_size', type=int, default=24)
    parser.add_argument('-length', type=int, default=16)
    parser.add_argument('-learnable', type=str, default='[0,1,1,1,1]')
    parser.add_argument('-niter', type=int)
    parser.add_argument('-use-gpu', type=int,default=1)
    parser.add_argument('-pretrain', type=str,default='./model/latest')
    parser.add_argument('-save-dir', type=str,default='./model')
    parser.add_argument('-epoch', type=int,default=150)
    parser.add_argument('-epoch_num', type=int,default=0)
    parser.add_argument('-phase', type=str,default='train')
    parser.add_argument('-model', type=str,default='./model/latest')
    parser.add_argument('-size', type=int,default=112)
    parser.add_argument('-dataset',type=str,default='hmdb',help ='ucf or hmdb')
    args = parser.parse_args()
    return args
def eval(args):
    place = fluid.CPUPlace() if not args.use_gpu else fluid.CUDAPlace(0) 
   
    with fluid.dygraph.guard(place):
        if args.dataset=='ucf':
             eval_model =MODEL.resnet_3d_v1(50,101,size=args.size,batch_size=args.batch_size)
        else:
             eval_model =MODEL.resnet_3d_v1(50,51,batch_size=args.batch_size,size=args.size)
        
        para_state_dict, _ = fluid.load_dygraph(args.model)
        eval_model.load_dict(para_state_dict)
        eval_model.train()
        print('开始预测')
    
        # eval_reader = reader.DS('RepNet/new_train.txt', 'work/data/', model='2d', mode='rgb', random=False,length=32, batch_size=args.batch_size,size=args.size).create_reader()
        eval_reader=reader.DS('RepNet/new_test.txt', 'work/data/', model='2d', mode='rgb', length=32, batch_size=args.batch_size,size=args.size).create_reader()
        acc_list = []
        for batch_id, data in enumerate(eval_reader()):
            dy_x_data = np.array([x[0] for x in data]).astype('float32')
            y_data = np.array([[x[1]] for x in data]).astype('int64')
          #  dy_x_data=random_noise(dy_x_data).astype('float32')
            img = fluid.dygraph.to_variable(dy_x_data)
            label = fluid.dygraph.to_variable(y_data)
            out = eval_model(img)
            acc = fluid.layers.accuracy(input=out, label=label)
            acc_list.append(acc.numpy()) 
            print(batch_id,'准确率:', acc.numpy())

        print("测试集准确率为:{}".format(np.mean(acc_list)))

def train(args):
    
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        
        if args.dataset=='ucf':
             train_model =MODEL.resnet_3d_v1(50,101,batch_size=args.batch_size,size=args.size)
        else:
             train_model =MODEL.resnet_3d_v1(50,51,batch_size=args.batch_size,size=args.size)
       # train_model =res.ResNet('resnet',50)
           # scale lr for flow layer
        # params = model.parameters()
        # params=train_model.parameters(include_sublayers=True)
        # params = [p for p in params]
        # other = []
        # print(len(params))
        # ln = eval(args.learnable)
        # if ln[0] == 1:
        #     other += [p for p in params if (p.sum() == train_model.module.flow_layer.img_grad.sum()).all() and p.size() == train_model.module.flow_layer.img_grad.size()]
        #     params = [p for p in params if (p.sum() != train_model.module.flow_layer.img_grad.sum()).all() or p.size() != train_model.module.flow_layer.img_grad.size()]
        #     params = [p for p in params if (p.sum() != train_model.module.flow_layer.img_grad2.sum()).all() or p.size() != train_model.module.flow_layer.img_grad2.size()]

        # if ln[1] == 1:
        #     other += [p for p in params if (p.sum() == train_model.module.flow_layer.f_grad.sum()).all() and p.size() == train_model.module.flow_layer.f_grad.size()]
        #     other += [p for p in params if (p.sum() == train_model.module.flow_layer.f_grad2.sum()).all() and p.size() == train_model.module.flow_layer.f_grad2.size()]
        #     params = [p for p in params if (p.sum() != train_model.module.flow_layer.f_grad.sum()).all() or p.size() != train_model.module.flow_layer.f_grad.size()]
        #     params = [p for p in params if (p.sum() != train_model.module.flow_layer.f_grad2.sum()).all() or p.size() != train_model.module.flow_layer.f_grad2.size()]

        # if ln[2] == 1:
        #     other += [p for p in params if (p.sum() == train_model.module.flow_layer.t.sum()).all() and p.size() == train_model.module.flow_layer.t.size()]
        #     params = [p for p in params if (p.sum() != train_model.module.flow_layer.t.sum()).all() or p.size() != train_model.module.flow_layer.t.size()]

        # if ln[3] == 1:
        #     other += [p for p in params if (p.sum() == train_model.module.flow_layer.l.sum()).all() and p.size() == train_model.module.flow_layer.l.size()]
        #     params = [p for p in params if (p.sum() != train_model.module.flow_layer.l.sum()).all() or p.size() != train_model.module.flow_layer.l.size()]

        # if ln[4] == 1:
        #     other += [p for p in params if (p.sum() == train_model.module.flow_layer.a.sum()).all() and p.size() == train_model.module.flow_layer.a.size()]
        #     params = [p for p in params if (p.sum() != train_model.module.flow_layer.a.sum()).all() or p.size() != train_model.module.flow_layer.a.size()]


            
        # #print([p for p in model.parameters() if (p == model.module.flow_layer.t).all()])
        # #print(other)
        # print(len(params), len(other))
        # #exit()

        # lr = 0.01
        # # solver = optim.SGD([{'params':params}, {'params':other, 'lr':0.01*lr}], lr=lr, weight_decay=1e-6, momentum=0.9)
        # # lr_sched = optim.lr_scheduler.ReduceLROnPlateau(solver, patience=7)
        # opt = fluid.optimizer.MomentumOptimizer(learning_rate=fluid.layers.natural_exp_decay(
        #     learning_rate=lr,
        #     decay_steps=10000,
        #     decay_rate=0.5,
        #     staircase=True),momentum=0.9, parameter_list=[{'params':params}, {'params':other, 'lr':0.01*lr}])     

        
        clip = fluid.clip.GradientClipByNorm(clip_norm=1.0)
      #  opt = fluid.optimizer.MomentumOptimizer(0.01, 0.9,use_nesterov =True,
       # regularization=fluid.regularizer.L2Decay(
        # regularization_coeff=1e-6),parameter_list=train_model.parameters())#,grad_clip=clip)
        opt = fluid.optimizer.AdamOptimizer(0.001, 0.9,parameter_list=train_model.parameters(), regularization=fluid.regularizer.L2Decay(
        regularization_coeff=1e-6),epsilon=1e-8)#,grad_clip=clip)

        for i in train_model.state_dict():
            print(i+' '+str(train_model.state_dict()[i].shape))
        if  args.pretrain:
            
            model, _ = fluid.dygraph.load_dygraph(args.pretrain)
            '''
            model['classify.weight']=np.repeat(model['classify.weight'],2,axis=1)[:,:101]
            model['classify.bias']=np.repeat(model['classify.bias'],2,axis=0)[:-1]
            
            model['rep_flow.conv4Ix.weight']=np.repeat( model['rep_flow.conv4Ix.weight'],16,axis=1)
            model['rep_flow.conv4Iy.weight']=np.repeat( model['rep_flow.conv4Iy.weight'],16,axis=1)
            model['rep_flow.conv4px.weight']=np.repeat( model['rep_flow.conv4px.weight'],16,axis=1)
            model['rep_flow.conv4py.weight']=np.repeat( model['rep_flow.conv4py.weight'],16,axis=1)
            model['rep_flow.conv4u.weight']=np.repeat( model['rep_flow.conv4u.weight'],16,axis=1)
            model['rep_flow.conv4v.weight']=np.repeat(model['rep_flow.conv4v.weight'],16,axis=1)
            model['rep_flow2.conv4Ix.weight']=np.repeat( model['rep_flow2.conv4Ix.weight'],16,axis=1)
            model['rep_flow2.conv4Iy.weight']=np.repeat( model['rep_flow2.conv4Iy.weight'],16,axis=1)
            model['rep_flow2.conv4px.weight']=np.repeat( model['rep_flow2.conv4px.weight'],16,axis=1)
            model['rep_flow2.conv4py.weight']=np.repeat( model['rep_flow2.conv4py.weight'],16,axis=1)
            model['rep_flow2.conv4u.weight']=np.repeat( model['rep_flow2.conv4u.weight'],16,axis=1)
            model['rep_flow2.conv4v.weight']=np.repeat(model['rep_flow2.conv4v.weight'],16,axis=1)'''
            train_model.load_dict(model)
        train_model.train()
        # build model
        if not os.path.exists(args.save_dir):
             os.makedirs(args.save_dir)

        # get reader
        if args.dataset=='ucf':
            train_reader = reader.DS('/home/aistudio/train.list', 'data/data48916/', model='2d', mode='rgb', length=32, batch_size=args.batch_size,size=args.size).create_reader()
        else:
            train_reader = reader.DS('RepNet/new_train.txt', 'work/data/', model='2d', mode='rgb', length=32, batch_size=args.batch_size,size=args.size).create_reader()     
        epochs = args.epoch 
        
       # lowest_loss=eval_to_select_best_model(train_model,1000)[1]
        lowest_loss=10
       
        log = []
        if args.dataset=='ucf':
            file='log_ucf.json'
        else :
            file='RepNet/log.json'
        with open(file,'r') as f:
            data=json.load(f)
            log=data
            
        for i in range(args.epoch_num,epochs):
            start=time.time()
            acc_list=[]
            loss_list=[]
            info={'epoch_num':None,'iterations':[], 'train_avg_loss':None, 'val_avg_loss':None, 'train_acc':None, 'val_acc':None}
            for batch_id, data in enumerate(train_reader()):
                
                dy_x_data = np.array([x[0] for x in data]).astype('float32')
                y_data = np.array([[x[1]] for x in data]).astype('int64')
               
               # dy_x_data=random_noise(dy_x_data).astype('float32')
                img = fluid.dygraph.to_variable(dy_x_data)
                label = fluid.dygraph.to_variable(y_data)
                label.stop_gradient = True
                
                compute_start=time.time()
                out = train_model(img)
                compute_end=time.time()
                acc = fluid.layers.accuracy(input=out, label=label)
                acc_list.append(acc.numpy())
                loss = fluid.layers.cross_entropy(out, label)
                avg_loss = fluid.layers.mean(loss)
                info['iterations'].append(avg_loss.numpy()[0].item())
                loss_list.append(avg_loss.numpy())
                avg_loss.backward()
                
               
                opt.minimize(avg_loss)
                train_model.clear_gradients()

                end=time.time()
                print("Loss at epoch {} step {}: loss:{}, acc: {},compute_time:{},total_time:{}"
                .format(i, batch_id, avg_loss.numpy(), acc.numpy(),compute_end-compute_start,end-start))    
                start=time.time()
                
            print('训练集正确率:{},训练集平均loss:{}'.format(np.mean(acc_list),np.mean(loss_list)))
            os.system('rm  ./model/latest.pdparams &&'+'ln -s /home/aistudio/model/rep_model_'+args.dataset+'_'+str(args.size)+'_'+str(i)+'.pdparams '+args.save_dir + '/latest.pdparams')        
              
            fluid.dygraph.save_dygraph(train_model.state_dict(), args.save_dir + '/rep_model_'+args.dataset+'_'+str(args.size)+'_'+str(i))
            
            _,loss,score= eval_to_select_best_model(train_model,lowest_loss)
            info['val_avg_loss']=loss.item()
            info['val_acc']=score.item()
            info['train_avg_loss']=np.mean(loss_list).item()
            info['train_acc']=np.mean(acc_list).item()
            info['epoch_num']=i
            log.append(info)
            if _:
             #   os.system('rm  ./model/best.pdparams &&'+'ln -s /home/aistudio/model/rep_model_'+args.dataset+'_'+str(args.size)+'_'+str(i)+'.pdparams '+args.save_dir + '/best.pdparams')        
                lowest_loss=loss
            else:
              #  model, _ = fluid.dygraph.load_dygraph(args.pretrain)
              #  train_model.load_dict(model)
                 pass
            train_model.train()
            with open(file,'w') as f:
                json.dump(log,f)
            
            
def eval_to_select_best_model(model,pre_loss):
        
        model.eval()
        if args.dataset=='ucf':
            eval_reader = reader.DS('/home/aistudio/val.list', 'data/data48916/', model='2d', mode='rgb',random=False, length=32, batch_size=args.batch_size,size=args.size).create_reader()
        else:
            eval_reader = reader.DS('RepNet/new_test.txt', 'work/data/', model='2d', mode='rgb',random=False, length=32, batch_size=args.batch_size,size=args.size).create_reader()
        acc_list = []
        loss_list=[]
        for batch_id, data in enumerate(eval_reader()):
            dy_x_data = np.array([x[0] for x in data]).astype('float32')
            y_data = np.array([[x[1]] for x in data]).astype('int64')
          #  dy_x_data=random_noise(dy_x_data).astype('float32')
            img = fluid.dygraph.to_variable(dy_x_data)
            label = fluid.dygraph.to_variable(y_data)
            out = model(img)
            acc = fluid.layers.accuracy(input=out, label=label)
            loss = fluid.layers.cross_entropy(out, label)
            avg_loss = fluid.layers.mean(loss)
            loss_list.append(avg_loss.numpy())
            acc_list.append(acc.numpy()) 
        score=np.mean(acc_list)
        loss=np.mean(loss_list)
        print("验证集准确率为:{},平均loss:{}".format(score,loss))
        if loss<pre_loss:
            return True,loss,score
        return False,loss,score           
if __name__ == "__main__":
    args = parse_args()
    
    if args.phase =='train':
        train(args)
    elif args.phase=='eval':
        eval(args)
