import  math
import cv2
import numpy as np
import scipy.signal as signal

cap = cv2.VideoCapture(-1)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

print(width, height)

ret, frame = cap.read()

def gaussian_filter(kernel_size, sigma=1, muu=-0.5):
    # Initializing value of x,y as grid of kernel size
    # in the range of kernel size
    x, y = np.meshgrid(np.linspace(-2, 2, kernel_size),np.linspace(-2, 2, kernel_size))
    dst = np.sqrt(x**2+y**2)
    # lower normal part of gaussian
    normal = 1/(2 * np.pi * sigma**2)
    # Calculating Gaussian filter
    gauss = np.exp(-((dst-muu)**2 / (2.0 * sigma**2))) * normal
    # normalize between 0 and 1
    gMax = gauss.max()
    for i in range(kernel_size):
        for j in range(kernel_size):
            gauss[i][j] = gauss[i][j]/gMax
    return gauss

def LKnextPoint(frame, lastFrame, size, centerPointX, centerPointY, gauss):
    
    #print(centerPointX, centerPointY)
    leftTopEdgeX,leftTopEdgeY,rightBottomEdgeX,rightBottomEdgeY = centerPointX-size,centerPointY-size,centerPointX+size,centerPointY+size
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[leftTopEdgeY:rightBottomEdgeY,leftTopEdgeX:rightBottomEdgeX]
    lastFrame = cv2.cvtColor(lastFrame, cv2.COLOR_BGR2GRAY)[leftTopEdgeY:rightBottomEdgeY,leftTopEdgeX:rightBottomEdgeX]

    frame = cv2.bilateralFilter(frame,7,50,50)

    grad_x = np.multiply(gauss,np.array(cv2.Sobel(frame, cv2.CV_32F, 1, 0, ksize=3)))
    grad_y = np.multiply(gauss,np.array(cv2.Sobel(frame, cv2.CV_32F, 0, 1, ksize=3)))
    grad_t = np.multiply(gauss,signal.convolve2d(lastFrame, np.array([[1., 1.], [1., 1.]]), boundary='symm', mode='same') 
        + signal.convolve2d(frame,np.array([[-1.,-1.],[-1.,-1.]]),boundary='symm',mode='same'))

    cv2.imshow("Gauss",gauss)

    A = np.asmatrix(np.array([[grad_x.flatten()],[grad_y.flatten()]])).transpose()
    b = np.array([grad_t.flatten()]).transpose()

    ASqr = A.T * A
    v = np.linalg.pinv(ASqr)*A.T*b

    return v

def DrawKernelOrArrow(frame,vX,vY,vWeight,centerPointX,centerPointY,size,drawKernel,drawArrow):
    if drawArrow == True:
        frame1 = cv2.arrowedLine(frame,(centerPointX,centerPointY),(int((vX*vWeight)+centerPointX),int((vY*vWeight)+centerPointY)),(0,0,255),2,8,0,0.1)
    if drawKernel == True:
        frame1 = cv2.rectangle(frame,(centerPointX-size,centerPointY-size),(centerPointX+size,centerPointY+size),(255,0,0),1)
    return frame1

class LKkernel:
    def __init__(self,size,centerPointX,centerPointY,gauss):
        self.size = size
        self.X = centerPointX
        self.Y = centerPointY
        self.gauss = gauss

    def LKpointMove(self,frame,lastFrame):
        if self.X > 640 - size:
            self.X = 640 - size - 2
        if self.Y > 480 - size:
            self.Y = 480 - size - 2
        if self.X < size:
            self.X = size + 2
        if self.Y < size:
            self.Y = size + 2
        self.v = LKnextPoint(frame,lastFrame,size,self.X,self.Y,self.gauss)
        self.X += int(self.v[0]*(size/10))
        self.Y += int(self.v[1]*(size/10))
        #return self.X,self.Y,self.v

    def getLKPointMoveX(self):
        return self.X
    
    def getLKPointMoveY(self):
        return self.Y
    
    def getLKPointMoveVX(self):
        return self.v[0]
    
    def getLKPointMoveVY(self):
        return self.v[1]

class PointListDrawAvg:
    def __init__(self,numPointsTrack,frame,size):
        self.ListOfPoints = []
        self.numPointsTrack = numPointsTrack
        self.frame = frame
        self.size = size
        self.gaussian = gaussian_filter(size*2)

    def startPoints(self,frame):
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        Roise= cv2.bilateralFilter(gray,7,50,50)
        corners = cv2.goodFeaturesToTrack(Roise,self.numPointsTrack,0.05,20)
        corners = np.int0(corners)
        for i in range(self.numPointsTrack):
            self.ListOfPoints.append(LKkernel(self.size,corners[i][0][0],corners[i][0][1],self.gaussian))

    def getLsPoints(self):
        return self.ListOfPoints

    def ResetPoint(self, point):
        self.ListOfPoints.remove(point)
        print("Hello")
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        Roise= cv2.bilateralFilter(gray,7,50,50)
        corners = cv2.goodFeaturesToTrack(Roise,1,0.05,20)
        corners = np.int0(corners)
        print("corners",corners)
        self.ListOfPoints.append(LKkernel(self.size,corners[0][0][0],corners[0][0][1],self.gaussian))

size = 20
points = 15
countFrames = False
while cap.isOpened():
    check, frame = cap.read()

    CopyedFrameForArrows = frame

    if(countFrames == False):
        a = PointListDrawAvg(points,frame,size)
        a.startPoints(frame)
        for i in a.getLsPoints():
            print(i)
        lastFrame = frame

    XmSum = 0
    YmSum = 0
    for i in a.getLsPoints():
        i.LKpointMove(frame,lastFrame)
        Xm = float(i.getLKPointMoveVX())
        Ym = float(i.getLKPointMoveVY())
        XmSum += Xm
        YmSum += Ym
        MoveX = i.getLKPointMoveX()
        MoveY = i.getLKPointMoveY()
        CopyedFrameForArrows = DrawKernelOrArrow(CopyedFrameForArrows,Xm,Ym,7,MoveX,MoveY,size,True,True)
        #print(Xm,i.getLKPointMoveY())
        if(MoveX > 640-size or MoveY > 480-size or MoveX < size or MoveY < size):
            a.ResetPoint(i)


    avg = math.sqrt((0-XmSum/points)**2+(0-YmSum/points)**2)
    direction = math.atan2((XmSum/points)-0,(YmSum/points)-0)
    CopyedFrameForArrows = DrawKernelOrArrow(CopyedFrameForArrows,XmSum/points,YmSum/points,7,320,240,0,False,True)
    CopyedFrameForArrows = DrawKernelOrArrow(CopyedFrameForArrows,XmSum/points,0,7,320,240,0,False,True)
    CopyedFrameForArrows = cv2.resize(CopyedFrameForArrows,(1000,800),interpolation=cv2.INTER_LINEAR)
    print((direction/math.pi)*180,avg)
    #$print(avg)

    XmSum = 0
    YmSum = 0

    lastFrame = frame
    cv2.imshow('ArrowFrame',CopyedFrameForArrows)
    countFrames = True

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()