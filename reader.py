import numpy as np
import random
import sys
sys.path.append('/home/aistudio/_lintel')
import os
#import lintel
import functools
import cv2
import paddle
from scipy import misc
from PIL import Image
dw=[]
class DS:

    def __init__(self, split_file, root, mode='rgb', length=16, model='2d', random=True, c2i={},batch_size=16,buf_size=1024,num_reader_threads=8,size=112):
        self.class_to_id = c2i
        self.id_to_class = []
        for i in range(len(c2i.keys())):
            for k in c2i.keys():
                if c2i[k] == i:
                    self.id_to_class.append(k)
        cid = 0
        self.data = []
        self.num_reader_threads=num_reader_threads
        self.model = model
        self.size = size
       
        self.buf_size=buf_size
        self.batch_size=batch_size
        self.filelist=split_file
        self.data=[]
        with open(split_file, 'r') as f:
            for l in f.readlines():
                if len(l) <= 5:
                    continue
                v,c = l.strip().split(' ')
                if c not in self.class_to_id:
                    self.class_to_id[c] = cid
                    self.id_to_class.append(c)
                    cid += 1
                self.data.append([os.path.join(root, v), self.class_to_id[c]])

        self.split_file = split_file
        self.root = root
        self.mode = mode
        self.length = length
        self.random = random

    


   

    def create_reader(self):
        _reader = self._reader_creator(self.filelist, 
                                       shuffle=self.random,
                                       num_threads=self.num_reader_threads,
                                       buf_size=self.buf_size)

        def _batch_reader():
            batch_out = []
            for imgs, label in _reader():
                if imgs is None:
                    continue
                batch_out.append((imgs, label))
                if len(batch_out) == self.batch_size:
                    yield batch_out
                    batch_out = []

        return _batch_reader

    def _reader_creator(self,
                        split_file,
                        shuffle=False,
                        num_threads=1,
                        buf_size=1024
                        ):
        def reader():
            with open(split_file, 'r') as f:
                lines = [line.strip() for line in f.readlines()]
                if shuffle:
                    random.shuffle(lines)
                for line in lines:
                    vid = line.strip()
                    yield vid
        def load_frames_from_video(vid_file,length):
            cap=cv2.VideoCapture(vid_file)
            total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            t=total-length*2
           
            import random
            if t>0:
               start=random.randint(0,t)
               step=2
            elif total>length:
                start=random.randint(0,total-length)
                step=1
            else:
                start=0
                step=1
               
            list_img=[]
            cap.set(cv2.CAP_PROP_POS_FRAMES,start)
            cap.grab()
            for i in range(length if length< total else total):
               
                success,frame=cap.read()
                if success and (start+i)%step==0:
                    frame=cv2.resize(frame,(480,480))
                    list_img.append(np.array(frame))
                if not success:
                    break        
            cap.release()
            res=length-len(list_img)
            for i in range(res):
                
                list_img.append(list_img[-1])
            
            return np.array(list_img)

        # def resize(group_img,size):
                
        #         df=[]
        #         for img in group_img:
        #             df.append(np.array(img))
        #         df=np.array(df)
        #         # print('shape',df.shape)
        #         # n,h,w,c=df.shape
        #         # w=w//2
        #         # h=h//2
        #         # if not self.random:
        #         #     i = int(round((h-self.size)/2.))
        #         #     j = int(round((w-self.size)/2.))
        #         #     df = np.reshape(df, newshape=(self.length, h*2, w*2, 3))[::2,::2,::2,:][:, i:-i, j:-j, :]
        #         # else:
        #         #     th = self.size
        #         #     tw = self.size
        #         #     # print(vid)
        #         #     # print(h,w,th,tw)#120 160 112 112
        #         #     i = random.randint(0, h - th) if h!=th else 0
        #         #     j = random.randint(0, w - tw) if w!=tw else 0
        #         #     # print(df.shape)
                    
        #         #     df = np.reshape(df, newshape=(self.length, h*2, w*2, 3))[::2,::2,::2,:][:, i:i+th, j:j+tw, :]
                    
        #         # newgroup=np.array(df)
        #         return df
        def resize(df,size):              
            n,h,w,c=df.shape
            df = np.frombuffer(np.array(df), dtype=np.uint8)
            w=w//2
            h=h//2
            # print("w,h",w,h,size)
            # center crop
            if not self.random:
                i = int(round((h-self.size)/2.))
                j = int(round((w-self.size)/2.))
                df = np.reshape(df, newshape=(self.length, h*2, w*2, 3))[::2,::2,::2,:][:, i:-i, j:-j, :]
            else:
                th = self.size
                tw = self.size
                # print(vid)
                # print(h,w,th,tw)#120 160 112 112
                i = random.randint(0, h - th) if h!=th else 0
                j = random.randint(0, w - tw) if w!=tw else 0
                # print(df.shape)
                
                df = np.reshape(df, newshape=(self.length, h*2, w*2, 3))[::2,::2,::2,:][:, i:i+th, j:j+tw, :]
            return df
        def retype(group_img):
            df=[]
            for img in group_img:
                df.append(np.array(img))
            df=np.array(df)
            return df
        def video_reader(vid):
             
            vid, cls = vid.split(' ')      
                    
            df=load_frames_from_video(self.root+vid,self.length)
            # if self.random:#数据增强加了之后需调小batch_size    
            #     df = group_multi_scale_crop(df, 240)
            #     df = group_random_crop(df, self.size)
            #     # df = group_random_flip(df)
            # else:
            #     df=group_center_crop(df,self.size)
            # df=retype(df)
            df=resize(df,self.size) 
            
            if self.mode == 'flow':
                #print(df[:,:,:,1:].mean())
                #exit()
                # only take the 2 channels corresponding to flow (x,y)
                df = df[:,:,:,1:]
                if self.model == '2d':
                    # this should be redone...
                    # stack 10 along channel axis
                    df = np.asarray([df[:10],df[2:12],df[4:14]]) # gives 3x10xHxWx2
                    df = df.transpose(0,1,4,2,3).reshape(3,20,self.size,self.size).transpose(0,2,3,1)
                
                    
            df = 1-2*(df.astype(np.float32)/255)      

            if self.model == '2d':
                # 2d -> return TxCxHxW
                return df.transpose([0,3,1,2]), cls
            # 3d -> return CxTxHxW
            return df.transpose([3,0,1,2]), cls
              
        mapper = functools.partial(
            video_reader )

        return paddle.reader.xmap_readers(mapper, reader, num_threads, buf_size)

        def __len__(self):
          return len(self.data)

#===================================================

def group_multi_scale_crop(img_group, target_size, scales=None, \
                           max_distort=1, fix_crop=True, more_fix_crop=True):                    
    scales = scales if scales is not None else [1, .875, .75, .66]
    input_size = [target_size, target_size]

    im_size = img_group[0].shape
    # print(im_size)

    # get random crop offset
    def _sample_crop_size(im_size):
        image_w, image_h = im_size[0], im_size[1]

        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in scales]
        crop_h = [
            input_size[1] if abs(x - input_size[1]) < 3 else x
            for x in crop_sizes
        ]
        crop_w = [
            input_size[0] if abs(x - input_size[0]) < 3 else x
            for x in crop_sizes
        ]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_step = (image_w - crop_pair[0]) / 4
            h_step = (image_h - crop_pair[1]) / 4

            ret = list()
            ret.append((0, 0))  # upper left
            if w_step != 0:
                ret.append((4 * w_step, 0))  # upper right
            if h_step != 0:
                ret.append((0, 4 * h_step))  # lower left
            if h_step != 0 and w_step != 0:
                ret.append((4 * w_step, 4 * h_step))  # lower right
            if h_step != 0 or w_step != 0:
                ret.append((2 * w_step, 2 * h_step))  # center

            if more_fix_crop:
                ret.append((0, 2 * h_step))  # center left
                ret.append((4 * w_step, 2 * h_step))  # center right
                ret.append((2 * w_step, 4 * h_step))  # lower center
                ret.append((2 * w_step, 0 * h_step))  # upper center

                ret.append((1 * w_step, 1 * h_step))  # upper left quarter
                ret.append((3 * w_step, 1 * h_step))  # upper right quarter
                ret.append((1 * w_step, 3 * h_step))  # lower left quarter
                ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

            w_offset, h_offset = random.choice(ret)

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    crop_w, crop_h, offset_w, offset_h = _sample_crop_size(im_size)
    crop_img_group = [
        Image.fromarray(img).crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
        for img in img_group
    ]
    ret_img_group = [
        img.resize((input_size[0], input_size[1]), Image.BILINEAR)
        for img in crop_img_group
    ]

    return ret_img_group


def group_random_crop(img_group, target_size):
    w, h = img_group[0].size
    th, tw = target_size, target_size

    assert (w >= target_size) and (h >= target_size), \
        "image width({}) and height({}) should be larger than crop size".format(w, h, target_size)

    out_images = []
    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)

    for img in img_group:
        if w == tw and h == th:
            out_images.append(img)
        else:
            out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

    return out_images


def group_random_flip(img_group):
    v = random.random()
    if v < 0.5:
        ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
        return ret
    else:
        return img_group


def group_center_crop(img_group, target_size):
    img_crop = []
    for img in img_group:
        w, h = img.size
        th, tw = target_size, target_size
        assert (w >= target_size) and (h >= target_size), \
            "image width({}) and height({}) should be larger than crop size".format(w, h, target_size)
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        img_crop.append(img.crop((x1, y1, x1 + tw, y1 + th)))

    return img_crop


def group_scale(imgs, target_size):
    resized_imgs = []
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.size
        if (w <= h and w == target_size) or (h <= w and h == target_size):
            resized_imgs.append(img)
            continue

        if w < h:
            ow = target_size
            oh = int(target_size * 4.0 / 3.0)
            resized_imgs.append(img.resize((ow, oh), Image.BILINEAR))
        else:
            oh = target_size
            ow = int(target_size * 4.0 / 3.0)
            resized_imgs.append(img.resize((ow, oh), Image.BILINEAR))

    return resized_imgs




def video_loader(frames, nsample, seglen, mode):
    videolen = len(frames)
    average_dur = int(videolen / nsample)

    imgs = []
    for i in range(nsample):
        idx = 0
        if mode == 'train':
            if average_dur >= seglen:
                idx = random.randint(0, average_dur - seglen)
                idx += i * average_dur
            elif average_dur >= 1:
                idx += i * average_dur
            else:
                idx = i
        else:
            if average_dur >= seglen:
                idx = (average_dur - seglen) // 2
                idx += i * average_dur
            elif average_dur >= 1:
                idx += i * average_dur
            else:
                idx = i

        for jj in range(idx, idx + seglen):
            imgbuf = frames[int(jj % videolen)]
            img = imageloader(imgbuf)
            imgs.append(img)

    return imgs


def mp4_loader(filepath, nsample, seglen, mode):
    cap = cv2.VideoCapture(filepath)
    videolen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sampledFrames = []
    for i in range(videolen):
        ret, frame = cap.read()
        # maybe first frame is empty
        if ret == False:
            continue
        img = frame[:, :, ::-1]
        sampledFrames.append(img)
    average_dur = int(len(sampledFrames) / nsample)
    imgs = []
    for i in range(nsample):
        idx = 0
        if mode == 'train':
            if average_dur >= seglen:
                idx = random.randint(0, average_dur - seglen)
                idx += i * average_dur
            elif average_dur >= 1:
                idx += i * average_dur
            else:
                idx = i
        else:
            if average_dur >= seglen:
                idx = (average_dur - 1) // 2
                idx += i * average_dur
            elif average_dur >= 1:
                idx += i * average_dur
            else:
                idx = i

        for jj in range(idx, idx + seglen):
            imgbuf = sampledFrames[int(jj % len(sampledFrames))]
            img = Image.fromarray(imgbuf, mode='RGB')
            imgs.append(img)

    return imgs
#===========================================================


if __name__ == '__main__':
  with paddle.fluid.dygraph.guard(paddle.fluid.CPUPlace()):
    DS = UCF
   
    dataseta = DS('/home/aistudio/train.list', 'data/data48916/', random=True,model='3d', mode='flow',size=224, length=16, batch_size=32).create_reader()
    dataseta1 = DS('/home/aistudio/train_hmdb.list', 'data/data47656/', random=True,model='3d', mode='flow',size=224, length=32, batch_size=16).create_reader()
 #   dataset = DS('data/hmdb/split1_test.txt', '/ssd/hmdb/', model='2d', mode='rgb', length=16, c2i=dataseta.class_to_id)
    for i,data in enumerate(dataseta()):
      if i==2:
         break
      
      x = np.array([x[0] for x in data]).astype('float32')
      y = np.array([[x[1]] for x in data])
      print(x.shape)
