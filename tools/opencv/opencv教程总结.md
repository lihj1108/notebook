#### 1.Gui Features in Opencv  Gui 功能

##### 1.Getting Started with Images  图像入门

imread()  读图片

imshow()  展示图片

imwrite()  保存图片

waitKey()  等待按键后再往下执行

destroyAllWindows()  删除所有窗口

##### 2.Getting Started with Videos  视频入门

pass

##### 3.Drawing Functions in OpenCV  绘图函数

line()  在图片上画直线

rectangle()  画矩形

circle()  画圆

ellipse()  画椭圆

polylines()  画多边形

putText()  在图片上写字

selectROI()  在图像上画出感兴趣区域，并返回区域坐标

##### 4.Mouse as a Paint-Brush  鼠标作为画笔

pass

##### 5.Trackbar as the Color Palette  轨迹栏作为调色板

pass

#### 2.Core Operations  核心操作

##### 1.Basic Operations on Images  基本操作

shape  获取图片的形状

item(h,w,c)  获取图片对应位置的像素值

itemset()  在图片对应位置设置指定的像素值

img[row1:row2, col1:col2]  获取指定区域的图像切片，也可以给指定区域的图片赋值

split()  将图片按最后一维进行拆分

merge()  合并多个维度

copyMakeBorder()  给图片创建边框

##### 2.Arithmetic Operations on Images  图像的算术运算

add()  加法

addWeighted()  两张图片按权重融合

bitwise_not()  位运算 非

bitwise_and()  位运算 与

##### 3.Performance Measurement and Improvement Techniques  绩效衡量和改进技术

pass

#### 3.Image Processing in OpenCV  图像处理

##### 1.Changing Colorspaces  改变色彩空间

inRange()  检查图片各像素点的像素值是否在设定的区间内

cvtColor()  图片模式转换

##### 2.Geometric Transformations of Images  几何变换

resize()  调整图片至指定的大小

getRotationMatrix2D()  计算二维旋转的仿射矩阵

getAffineTransform()  计算仿射变换矩阵，需要知道原图和变换后的图三个点的对应坐标

warpAffine()  仿射变换，原始图像中的所有平行线在输出图像中仍然是平行的，是二维坐标变换

getPerspectiveTransform()  计算透视变换矩阵，需要原图和变换后的图四个点的对应坐标

warpPerspective()  透视变换，原始图像中的所有平行线在输出图像中可能不平行，是三维坐标变换

##### 3.Image Thresholding  图像阈值

threshold()  图片像素阈值筛选

adaptiveThreshold()  自适应阈值筛选

##### 4.Smoothing Images  图像平滑

filter2D()  用2维卷积核对图像进行均值平滑

blur()  均值平滑

GaussianBlur()  高斯平滑

medianBlur()  中位数平滑

bilateralFilter()  双边过滤，在保留边缘的同时进行图像平滑

##### 5.Morphological Transformations  形态变换

erode()  腐蚀，让边缘变窄

dilate()  膨胀，让边缘变厚

morphologyEx()  opening，先侵蚀后膨胀；closing，先膨胀后侵蚀

##### 6.Image Gradients  图像梯度

Sobel()  使用Sobel方法对图片的像素进行求导，可以指定求导的方向，x轴或y轴

Scharr()  使用Scharr方法对图片的像素进行求导，可以指定求导的方向，x轴或y轴

Laplacian()  使用Laplacian方法对图片的像素进行求导

##### 7.Canny Edge Detection  边缘检测

Canny()  用Canny方法进行边缘检测

##### 8.Image Pyramids  图像金字塔

pyrDown()  图像均衡+下采样

pyrUp()  图像上采样+均衡

##### 9.Contours in OpenCV  图像轮廓

findContours()  计算图片轮廓，返回的轮廓上个点的坐标

drawContours()  在图像上画出轮廓点

contourArea()  计算轮廓的面积

arcLength()  计算轮廓的周长

approxPolyDP()  用一个曲线拟合轮廓，减少轮廓上的点

convexHull()  用一条凸曲线拟合轮廓

isContourConvex()  检查轮廓是否是凸曲线

boundingRect()  计算2D点集的矩形边界框，返回边界框左上角坐标，宽和高，边界框的两边分别是水平和垂直的

minAreaRect()  计算2D点集的最小面积矩形边界框，所以该矩形可能旋转矩形

boxPoints()  计算旋转矩形的四个顶点，用以绘制

minEnclosingCircle()  计算2D点集最小包络圆

fitEllipse()  用一个椭圆拟合一组2D点集

fitLine()  用一条直线拟合一组2D点集

findNonZero()  返回非零像素点的位置坐标

minMaxLoc()  返回像素点的最小值和最大值，以及它们的位置坐标

convexityDefects()  计算轮廓的凸缺陷，返回缺陷的起点、终点、最远点、到最远点的大概距离

pointPolygonTest()  计算某点与轮廓的最短距离，如果返回负数，则表示点在轮廓内

matchShapes()  计算两个轮廓的相似度，越小越相似

##### 10.Histograms in OpenCV  图像直方图

calcHist()  计算图片直方图，一维直方图只考虑灰度值，横轴是像素值的范围，纵轴是对应像素点的个数；二维直方图的输入是hsv格式的图片，考虑色相和饱和度

equalizeHist()  直方图均衡，将像素值较高的比较集中的区域均衡至[0, 255]范围内

createCLAHE().apply()  对比度限制的自适应直方图均衡

calcBackProject()  直方图反向投影，用于图像分割。先计算感兴趣区域的直方图，然后反向投影到原图像，原图像中和感兴趣区域相似的区域像素值将变为255，其他区域则变为0，达到图像分割的目的。

##### 11.Image Transforms in OpenCV  图像变换

dft()  计算图像的傅里叶变换，傅里叶变换用于查找频域。图像中的高频分量，指的是图像强度（亮度、灰度）变化剧烈的地方，主要是对图像边缘和轮廓的度量；低频分量指的是图像强度变换平缓的地方，主要是对政府图像的强度的综合度量。

idft()  进行傅里叶逆变换

magnitude()  计算傅里叶变换或逆傅里叶变换后的实部和虚部的平方和的平方根，用于显示图像，因为虚部无法展示

##### 12.Template Matching  模板匹配

matchTemplate()  模板匹配，在大图像中查找小图像

##### 13.Hough Line Transform  霍夫线变换

HoughLines()  使用标准霍夫变换在图像中查找直线

HoughLinesP()  使用概率霍夫变换在图像中查找直线

##### 14.Hough Circle Transform  霍夫圆变换

HoughCircles()  使用霍夫变换在图像中查找圆

##### 15.Image Segmentation with Watershed Algorithm  图像分割

distanceTransform()  计算灰度图中的像素点距离最近0像素点的距离

connectedComponents()  计算图像连通域

connectedComponentsWithStats()  计算图像连通域，并返回bbox信息

watershed()  使用分水岭算法基于图像连通域标记进行图像分割

##### 16.Interactive Foreground Extraction using GrabCut Algorithm  前景提取

grabCut()  用grabCut算法提取图像前景区域，用于图像分割

#### 4.Feature Detection and Description  特征检测和表述

##### 1.Understanding Features  理解图像特征

pass

##### 2.Harris Corner Detection  角点检测

cornerHarris()  使用harris方法进行角点检测

cornerSubPix()  优化角点的位置

##### 3.Shi-Tomasi Corner Detector & Good Features to Track  强角点检测

goodFeaturesToTrack()  查找图像上的强角点

##### 4.Introduction to SIFT (Scale-Invariant Feature Transform)  sift关键点检测

SIFT_create()  创建sift对象，可用于检测关键点

sift.detect()  检测关键点

sift.detectAndCompute()  检测关键点，并返回关键点的描述

drawKeypoints()  在图像上画出关键点

##### 5.Introduction to SURF (Speeded-Up Robust Features)  surf关键点检测

pass

##### 6.FAST Algorithm for Corner Detection  fast检测关键点

FastFeatureDetector_create()  创建fast特征检测器对象

fast.detect()  检测关键点

##### 7.BRIEF (Binary Robust Independent Elementary Features)

pass

##### 8.ORB (Oriented FAST and Rotated BRIEF)  orb关键点检测

ORB_create()  创建orb特征检测器对象

orb.detect()  检测关键点

orb.compute()  检测关键点，并返回关键点的描述

orb.detectAndCompute()  检测关键点，并返回关键点的描述

##### 9.Feature Matching  特征匹配

BFMatcher()  创建Brute-Force特征匹配器对象

bf.match()  对两个图像的关键点进行特征匹配

bf.knnMatch()  对两个图像上的关键点进行knn特征匹配

drawMatches()  在图像上画出匹配到的特征

drawMatchesKnn()  在图像上画出knn匹配到的特征

FlannBasedMatcher()  创建基于flann的特征匹配器对象

##### 10.Feature Matching + Homography to find Objects  特征匹配+单应性来查找目标

findHomography()  根据两个图片上匹配到的特征点来计算透视变换

perspectiveTransform()  执行向量的透视变换矩阵

#### 5.Video analysis(video module)

pass

#### 6.Camera Calibration and 3D Reconstruction

pass

#### 7.Machine Learning

pass

#### 8.Computational Photography

##### 1.Image Denoising

fastNlMeansDenoising()  对灰度图进行非局部均值去噪

fastNlMeansDenoisingColored()  对彩图进行非局部均值去噪

fastNlMeansDenoisingMulti()  对多张连续的灰度图进行非局部均值去噪

fastNlMeansDenoisingColoredMulti()  对多张连续的彩图进行非局部均值去噪

##### 2.Image Inpainting

inpaint()  图像修复，输入待修复图像和噪声mask

##### 3.High Dynamic Range (HDR)

pass

#### 9.Object Detection (objdetect module)

pass

