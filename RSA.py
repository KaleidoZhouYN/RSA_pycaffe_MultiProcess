import numpy as np

import cv2
from skimage import transform as transform
import nms

caffe_root = '../CaffeMex_v2/python'  #change to your caffe root path
import sys
sys.path.append(caffe_root)

import caffe

class RSA(object):
    """
        Description
    """
    
    def __init__(self,
                max_img = 2048.0,
                target_ratio = -4.0,
                det_thresh = 5,
                plot_thresh = 7,
                gpu_id = 2,
                anchor_scale = 1,
                factor = 1,
                anchor_box = (-44.7548,-44.7548,44.7548,44.7548),
                thresh_cls = 3,
                stride = 16,
                anchor_center = 7.5,
                anchor_pts = (-0.1719,-0.2204,0.1719,-0.2261,-0.0017,-0.0047,-0.1409,0.2034,0.1409,0.1978),
                nms_thres = 0.2,
                nms_score = 8,
                scale = (0,1,2,3,4)):
        self.max_img = max_img
        self.target_ratio = target_ratio
        self.det_thresh = det_thresh
        self.plot_thresh = plot_thresh
        self.gpu_id = gpu_id
        
        #load param form param.mat
        self.anchor_scale = anchor_scale
        self.factor = factor
        self.anchor_box = anchor_box
        self.thresh_cls = thresh_cls
        self.stride = stride
        self.anchor_center = anchor_center
        self.anchor_pts = anchor_pts
        self.nms_thres = 0.2
        self.nms_score = 8
        
        #model path and prototxt path
        model_1 = 'output/ResNet_3b_s16/tot_wometa_1epoch.caffemodel'
        proto_1 = 'model/res_pool2/test.prototxt'
        
        model_2 = 'output/ResNet_3b_s16_fm2fm_pool2_deep/65w.caffemodel'
        proto_2 = 'model/ResNet_3b_s16_fm2fm_pool2_deep/test.prototxt'
        
        model_3 = 'output/ResNet_3b_s16/tot_wometa_1epoch.caffemodel'
        proto_3 = 'model/ResNet_3b_s16_f2r/test.prototxt'
        
        #load caffe model
        caffe.set_mode_gpu()
        caffe.set_device(self.gpu_id)
        
        self.Net1 = caffe.Net(proto_1,model_1,caffe.TEST)
        self.Net2 = caffe.Net(proto_2,model_2,caffe.TEST)
        self.Net3 = caffe.Net(proto_3,model_3,caffe.TEST)
               
        
        #downsample scale
        self.scale_u = scale #(0,1,2,3,4)
        self.scale = np.power(2,self.scale_u)

        
    def gen_featmap(self,img):
         
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        #img = img[:,:,[2,1,0]] # BGR2RGB

        col,row,c = img.shape
        scale = self.max_img/max(col,row)        
        img = cv2.resize(img,(int(round(row*scale)),int(round(col*scale))))-127.0
        img = img.transpose(2,0,1)[np.newaxis,:,:,:] #reshpae to NCHW
        
        self.Net1.blobs['data'].reshape(*img.shape)
        np.copyto(self.Net1.blobs['data'].data,img)
        out = self.Net1.forward()['res2b']
        #out shape [1,C,H,W]
        
        return out
        
    def featmap_transfer(self,featmap):
        featmap_t = []
        ori_scale = 0
        inmap = featmap.copy()
        for i in range(len(self.scale_u)):
            scale_t = self.scale_u
            if i == 0:
                diffcnt = scale_t[i] - ori_scale
            else:
                diffcnt = scale_t[i] - scale_t[i-1]
            
            for cnt in range(diffcnt):
                self.Net2.blobs['data'].reshape(*inmap.shape)
                np.copyto(self.Net2.blobs['data'].data, inmap)
                inmap = self.Net2.forward()['res2b_trans_5']
            
            featmap_t.append(inmap.copy())
        return featmap_t
    
    def featmap_2_result(self,featmap_t,img):
        mins = 3
        maxs = 14
        num = len(featmap_t)
        parsed = {}
        parsed['cls'] = []
        parsed['pts'] = []
        parsed['box'] = []
        s = max(img.shape[0],img.shape[1])/self.max_img
        for id in range(num):
            #scale,s:resize pts and box from max_img to img_size
            flag,result = self.detect_all_by_featmap(featmap_t[id],self.scale[id]*s)
            if (flag != 0):
                
                parsed['cls'] += result['cls']
                parsed['pts'] += result['pts']
                parsed['box'] += result['box']
                
        return parsed
                
    def detect_all_by_featmap(self,featmap,scale):
        tmp = {}
        tmp['cls']=[]
        tmp['pts']=[]
        tmp['box']=[]
        flag = 0
        self.Net3.blobs['res2b'].reshape(*featmap.shape)
        np.copyto(self.Net3.blobs['res2b'].data,featmap)
        self.Net3.forward()
        # reshape back to [H*W]
        cls = self.Net3.blobs['rpn_cls'].data.squeeze()
        pts = self.Net3.blobs['rpn_reg'].data.squeeze() 
        
        anchor_box_len = (self.anchor_box[2]-self.anchor_box[0],self.anchor_box[3]-self.anchor_box[1])
        y,x = np.where(cls >= self.thresh_cls)
        
        for id in range(len(y)):
            #get pts
            anchor_center_now = ((x[id])*self.stride + self.anchor_center, (y[id])*self.stride + self.anchor_center)
            anchor_points_now = np.multiply(self.anchor_pts , anchor_box_len[0]) + np.matlib.repmat(anchor_center_now,1,5)
            pts_delta = np.multiply(pts[:,y[id],x[id]],anchor_box_len[0])
            pts_out = (pts_delta+anchor_points_now)
            
            tmp['cls'].append(cls[y[id],x[id]])
            #resize box & point to max_img
            point = np.array(pts_out).squeeze()
            box = self.get_rect_from_pts(point)
            tmp['pts'].append(np.multiply(point,scale))
            tmp['box'].append(np.multiply(box,scale))
            flag = 1
        return flag,tmp   
    
    def get_rect_from_pts(self,pts):
        std_pts = np.array([0.2,0.2,0.8,0.2,0.5,0.5,0.3,0.75,0.7,0.75]).reshape(-1,2)
        pts_ = np.array(pts).reshape(-1,2)
        similarity_t = transform.SimilarityTransform()
        assert(similarity_t.estimate(std_pts,pts_) == True,'gen rect from pts error')
        T_matrix = np.array(similarity_t.params)
        std_ptc = np.array([0.5,0.5,1]).reshape(3,-1)
        std_ptl = np.array([0,0,1]).reshape(3,-1)
        std_ptr = np.array([1,0,1]).reshape(3,-1)
        
        ptc = np.dot(T_matrix , std_ptc)
        ptl = np.dot(T_matrix , std_ptl)
        ptr = np.dot(T_matrix , std_ptr)
                
        w = ((ptl[0]-ptr[0])**2+(ptl[1]-ptr[1])**2)**(1.0/2)
        rect = np.round(np.array([ptc[0]-w/2,ptc[1]-w/2,ptc[0]+w/2,ptc[1]+w/2])).reshape(1,4)
        return rect
    
    def nms_parsed(self,parsed):
        box = np.vstack(parsed['box'])
        cls = np.hstack(parsed['cls'])
        pts = np.vstack(parsed['pts'])
        boxes = np.hstack([box, np.array(cls, ndmin=2).T])

        boxes,idx = nms.non_max_suppression(boxes,self.nms_thres)
        f_idx = boxes[:,4] > self.nms_score
        return cls[idx[f_idx]],box[idx[f_idx]],pts[idx[f_idx]]
        
    #return list  cls,box&pts
    def detect(self,img):
    #can only deal with one image
        featmap = self.gen_featmap(img)
        featmaps = self.featmap_transfer(featmap)
        parsed = self.featmap_2_result(featmaps,img)
        cls,box,pts = self.nms_parsed(parsed)
        return cls,box,pts
            