from paddle.fluid.dygraph import Conv3D,BatchNorm,Conv2D
from paddle import fluid
from paddle.fluid import layers
import paddle
import numpy as np
import gc
class FlowLayer(fluid.dygraph.Layer):
    def __init__(self,batch_size, channels=1, bottleneck=32, params=[0,1,1,1,1], n_iter=20,last=False,bn=True):
        super(FlowLayer,self).__init__()  
        self.batch_size=batch_size
        self.bottleneck = Conv2D(channels, bottleneck, stride=1, padding=0, filter_size=1,
                                 bias_attr=fluid.ParamAttr(trainable=False))
        
        self.unbottleneck = Conv2D(bottleneck*2,channels, stride=1, padding=(1,1), filter_size=(3,3),
                                bias_attr=fluid.ParamAttr(trainable=False))
        self.bn = BatchNorm(channels) if bn else None
        channels = bottleneck
        
        self.conv4Ix=Conv2D(channels,channels,padding=0,stride=1,filter_size=3,param_attr=fluid.ParamAttr(learning_rate=0.01,
                          initializer=fluid.initializer.NumpyArrayInitializer(np.array([[[[-0.5,0,0.5]]]*channels]*channels)),
                          trainable=params[0]==1), bias_attr=fluid.ParamAttr(trainable=False),groups=1)
        
        self.conv4Iy=Conv2D(channels,channels,padding=0,stride=1,filter_size=3,param_attr=fluid.ParamAttr(learning_rate=0.01,
                          initializer=fluid.initializer.NumpyArrayInitializer(np.array([[[[-0.5],[0],[0.5]]]*channels]*channels)),
                          trainable=params[0]==1), bias_attr=fluid.ParamAttr(trainable=False),groups=1)
        
        self.conv4px=Conv2D(channels,channels,padding=0,stride=1,filter_size=(1,2),param_attr=fluid.ParamAttr(learning_rate=0.01,
                          initializer=fluid.initializer.NumpyArrayInitializer(np.array([[[[-1,1]]]*channels]*channels)),
                          trainable=params[1]==1), bias_attr=fluid.ParamAttr(trainable=False),groups=1)
        
        self.conv4py=Conv2D(channels,channels,padding=0,stride=1,filter_size=(2,1),param_attr=fluid.ParamAttr(learning_rate=0.01,
                          initializer=fluid.initializer.NumpyArrayInitializer(np.array([[[[-1],[1]]]*channels]*channels)),
                          trainable=params[1]==1), bias_attr=fluid.ParamAttr(trainable=False),groups=1)
        
        self.conv4u=Conv2D(channels,channels,padding=0,stride=1,filter_size=(1,2),param_attr=fluid.ParamAttr(learning_rate=0.01,
                          initializer=fluid.initializer.NumpyArrayInitializer(np.array([[[[-1,1]]]*channels]*channels)),
                          trainable=params[1]==1), bias_attr=fluid.ParamAttr(trainable=False),groups=1)
        
        self.conv4v=Conv2D(channels,channels,padding=0,stride=1,filter_size=(2,1),param_attr=fluid.ParamAttr(learning_rate=0.01,
                          initializer=fluid.initializer.NumpyArrayInitializer(np.array([[[[-1],[1]]]*channels]*channels)),
                          trainable=params[1]==1), bias_attr=fluid.ParamAttr(trainable=False),groups=1)
        

        self.n_iter = n_iter           
        self.channels = channels
        
        self.theta = layers.create_parameter(shape=[1], dtype='float32',attr=fluid.ParamAttr(learning_rate=0.01,
                          initializer=fluid.initializer.NumpyArrayInitializer(np.array([0.3])),trainable=params[2]==1))
        self.lamda = layers.create_parameter(shape=[1], dtype='float32',attr=fluid.ParamAttr(learning_rate=0.01,
                          initializer=fluid.initializer.NumpyArrayInitializer(np.array([0.15])),trainable=params[3]==1))
        self.tau   = layers.create_parameter(shape=[1], dtype='float32',attr=fluid.ParamAttr(learning_rate=0.01,
                          initializer=fluid.initializer.NumpyArrayInitializer(np.array([0.25])),trainable=params[4]==1))       
        
 

    def norm_img(self, x):
        mx = layers.reduce_max(x)
        mn = layers.reduce_min(x)
        x = 255*(x-mn)/(mn-mx)
       
        return x
            
    def forward_grad(self, x):
        grad_x = self.conv4u(layers.pad(x, (0,0,0,0,0,0,0,1)))
        tmp = layers.unstack(grad_x,axis=2)
        tmp[-1]=tmp[-1]-tmp[-1]  #tmp[-1]=0
       
        grad_x=layers.stack(tmp,axis=2)
      
        
        grad_y = self.conv4v(layers.pad(x, (0,0,0,0,0,1,0,0)))
     
        tmp = layers.unstack(grad_y,axis=2)
        tmp[-1]=tmp[-1]-tmp[-1]   # tmp[-1]=0
        grad_y=layers.stack(tmp,axis=2)
        return grad_x, grad_y


    def divergence(self, x, y):
        
        tx = layers.pad(x[:,:,:,:], (0,0,0,0,0,0,1,0))
        ty = layers.pad(y[:,:,:,:], (0,0,0,0,1,0,0,0))
        grad_x = self.conv4px(tx)
        grad_y = self.conv4py(ty)
     
        return grad_x + grad_y
        
        
    def forward(self, x):
        
        '''
        bt,c,w,h=x.shape
        tmp=layers.reshape(x,shape=[48,-1,c,w,h])
        res=layers.reshape(tmp[:,:-1],shape=[-1,c,w,h])'''
        x = self.bottleneck(x)
        inp = self.norm_img(x)
        bt,c,w,h=inp.shape
        inp=layers.reshape(inp,shape=[self.batch_size,-1,c,w,h])
       
        x=inp[:,:-1]
        y=inp[:,1:]
       
        x = layers.reshape(layers.transpose(x,perm=[0,2,1,3,4]),shape=[-1,c,h,w])
        y = layers.reshape(layers.transpose(y,perm=[0,2,1,3,4]),shape=[-1,c,h,w])
        u1=fluid.dygraph.to_variable(np.zeros(x.shape)).astype('float32')
        u2=fluid.dygraph.to_variable(np.zeros(x.shape)).astype('float32')

        l_t = self.lamda * self.theta
        taut = self.tau/(self.theta+1e-12)
        
        grad2_x = self.conv4Ix(layers.pad(y,(0,0,0,0,0,0,1,1)))
        
        tmp = layers.unstack(grad2_x,axis=3)
        tmp[-1]=0.5 * (x[:,:,:,-1] - x[:,:,:,-2])
        tmp[0]=0.5 * (x[:,:,:,1] - x[:,:,:,0])
        grad2_x=layers.stack(tmp,axis=3)
     

        
        grad2_y = self.conv4Iy(layers.pad(y, (0,0,0,0,1,1,0,0)))
        tmp = layers.unstack(grad2_y,axis=2)
        tmp[-1]= 0.5 * (x[:,:,-1,:] - x[:,:,-2,:])
        tmp[0]=0.5 * (x[:,:,1,:] - x[:,:,0,:])
        grad2_y=layers.stack(tmp,axis=2)
       
    
        p11=fluid.dygraph.to_variable(np.zeros(x.shape)).astype('float32')
        p12=fluid.dygraph.to_variable(np.zeros(x.shape)).astype('float32')
        p21=fluid.dygraph.to_variable(np.zeros(x.shape)).astype('float32')
        p22=fluid.dygraph.to_variable(np.zeros(x.shape)).astype('float32')
        
        gsqx = grad2_x**2
        gsqy = grad2_y**2
       
        grad = gsqx + gsqy + 1e-12
     
        rho_c = y - grad2_x * u1 - grad2_y * u2 - x
        for i in range(self.n_iter):
            rho = rho_c + grad2_x * u1 + grad2_y * u2 + 1e-12
          
            mask1 = (rho < -l_t*grad).detach().astype('float32')
            mask1.stop_gradient = True
            tmp1=l_t * grad2_x
            tmp2=l_t * grad2_y
            v1=tmp1*mask1
            v2=tmp2*mask1
            
            mask2 = (rho > l_t*grad).detach().astype('float32')
            mask2.stop_gradient = True
            v1= -tmp1*mask2+v1
            v2=-tmp2*mask2+v2
           
            mask3=fluid.layers.ones(x.shape, dtype='float32')-(mask1+mask2-mask1*mask2)
            mask3.stop_gradient = True
            tmp1=(-rho/grad) * grad2_x
            tmp2=(-rho/grad) * grad2_y
           
            v1= tmp1*mask3+v1
            v2= tmp2*mask3+v2
            
            del rho
            del mask1
            del mask2
            del mask3

            v1 += u1
            v2 += u2
            
            u1 = v1 + self.theta * self.divergence(p11, p12)
            u2 = v2 + self.theta * self.divergence(p21, p22)
         
            del v1
            del v2
            u1 = u1
            u2 = u2
            
            u1x, u1y = self.forward_grad(u1)
            u2x, u2y = self.forward_grad(u2)
            
            p11 = (p11 + taut * u1x) / (1. + taut * layers.sqrt(u1x**2 + u1y**2 + 1e-12))
            p12 = (p12 + taut * u1y) / (1. + taut * layers.sqrt(u1x**2 + u1y**2 + 1e-12))
            p21 = (p21 + taut * u2x) / (1. + taut * layers.sqrt(u2x**2 + u2y**2 + 1e-12))
            p22 = (p22 + taut * u2y) / (1. + taut * layers.sqrt(u2x**2 + u2y**2 + 1e-12))
            del u1x
            del u1y
            del u2x
            del u2y
           

   
        flow = layers.concat([u1,u2], axis=1)
    
      #  flow = layers.transpose(layers.reshape(flow,shape=[b,t,c*2,h,w]),perm=[0,2,1,3,4])
        flow = self.unbottleneck(flow)
        flow = self.bn(flow) if self.bn else flow
        return  flow