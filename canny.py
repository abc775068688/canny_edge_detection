import cv2
import  numpy as np
import  math

def canny(img,threshodl1,threshodl2):
    #步骤1 高斯滤波
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    """
    cv2.cvtColor(p1,p2) 是颜色空间转换函数，p1是需要转换的图片，p2是转换成何种格式。
    cv2.COLOR_BGR2RGB 将BGR格式转换成RGB格式
    cv2.COLOR_BGR2GRAY 将BGR格式转换成灰度图片
    """
    new_gray=cv2.GaussianBlur(gray,(5,5),1) # 5*5大小模板 标准差是1
    gaussion_result=np.uint8(np.copy(new_gray))
    cv2.imshow('gaussian',gaussion_result)
    #步骤2 算梯度幅值
    W1,H1=new_gray.shape[:2]
    dx=np.zeros([W1-1,H1-1])
    dy=np.zeros([W1-1,H1-1])
    d=np.zeros([W1-1,H1-1])
    dgree=np.zeros([W1-1,H1-1])
    #sobel算子
    """
    1 0 -1
    2 0 -2  x方向
    1 0 -1
    
    1  2  1
    0  0  0 y方向
    -1 -2 -1
    """
    for i in range(1,W1-1):
        for j in range(1,H1-1):
            dx[i,j]=new_gray[i-1,j-1]+2*new_gray[i,j-1]+new_gray[i+1,j-1]-new_gray[i-1,j+1]-2*new_gray[i,j+1]-new_gray[i+1,j+1]
            dy[i,j]=new_gray[i-1,j-1]+2*new_gray[i-1,j]+new_gray[i-1,j+1]-new_gray[i+1,j-1]-2*new_gray[i+1,j]-new_gray[i+1,j+1]
            d[i,j]=np.sqrt(np.square(dx[i,j])+np.square(dy[i,j])) #图像梯度值作为图像强度值
            dgree[i,j]=math.degrees(math.atan2(dy[i,j],dx[i,j])) #计算梯度方向 把弧度值转换为角度  相当于 arctan(dy/dx)
            if dgree[i,j]<0:   #如果角度为负 加上360度
                dgree+=360

    d_r=np.uint8(np.copy(d))
    cv2.imshow('grad',d_r) #梯度值的图像显示出来

   #步骤3 非极大值抑制 not max stric
    W2,H2=d.shape
    NMS=np.copy(d)
    NMS[0,:]=NMS[:,H2-1]=NMS[W2-1,:]=NMS[:0]=0 #把最外层的值全部赋0
    """ 把所有线段分为八个部分 0 45 90 135 180 225 270 315 360
    根据梯度方向判断所属的区域 比较该点与梯度方向上相邻两点的值 如果比相邻点值都大则保留 否则置0
    """
    for i in range(1,W2-1):
        for j in range(1,H2-1):
            if dgree[i,j]==0:
                NMS[i,j]=0
            else:
                g1=None
                g2=None
                if (dgree[i,j]<22.5 or dgree[i,j]>=337.5) or (dgree[i,j]>=157.5 and dgree[i,j]<202.5):
                    g1=NMS[i,j-1]
                    g2=NMS[i,j+1]
                elif(dgree[i,j]<67.5 and dgree[i,j]>=22.5) or (dgree[i,j]>=202.5 and dgree[i,j]<247.5):
                    g1=NMS[i-1,j+1]
                    g2=NMS[i+1,j-1]
                elif(dgree[i,j]<112.5 and dgree[i,j]>=67.5) or (dgree[i,j]>=247.5 and dgree[i,j]<292.5):
                    g1=NMS[i-1,j]
                    g2=NMS[i+1,j]
                else:
                    g1=NMS[i-1,j-1]
                    g2=NMS[i+1,j+1]
                if NMS[i,j]<g1 or NMS[i,j]<g2:
                    NMS[i,j]=0


    nums_r=np.uint8(np.copy(NMS))
    cv2.imshow('nms',nums_r)

    # 双阈值算法检测 连接边缘
    W3,H3=NMS.shape
    DT=np.zeros([W3,H3],dtype=np.uint8)
    # 定义高阈值
    TL=min(threshodl1,threshodl2)
    TH=max(threshodl1,threshodl2)
    for i in  range(1,W3-1):
        for j in range(1,H3-1):
            if NMS[i,j]<TL: #小于低阈值则赋值为 0
                DT[i,j]=0
            elif NMS[i,j]<TH:#大于高阈值则赋值为 255最大灰度值
                DT[i,j]=255

            else: #判断相邻八点阈值 只要有一点阈值大于高阈值 我们则可以认为该点是我们检测到的边缘点
                if NMS[i-1,j]>TH or NMS[i-1,j+1]>TH or NMS[i-1,j-1]>TH or NMS[i,j-1]>TH or NMS[i,j+1]>TH or NMS[i+1,j-1]>TH \
                    or NMS[i+1,j]>TH  or NMS[i+1,j+1]>TH:
                    DT[i,j]=255

    return  DT

img=cv2.imread('0_1283586269IN6y.gif')
cv2.imshow('src',img)
result=canny(img,40,100)
cv2.imshow('dst',result)
cv2.waitKey(0)










