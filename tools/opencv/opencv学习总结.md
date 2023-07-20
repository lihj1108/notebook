##### 1.图片加载和保存

- cv2.imread(filename, flags) ：读取加载图片
- cv2.imshow(winname, mat) ： 显示图片
- cv2.waitKey(delay) ： 等待图片的关闭。0是等待按键关闭，>0是等待多少ms自动关闭
- cv2.imwrite(filename, img) ： 保存图片

##### 2.创建和销毁窗口

- cv2.namedWindow(winname, 属性)：创建一个窗口
- cv2.destroyWindow(winname)：销毁某个窗口
- cv2.destroyAllWindows()：销毁所有窗口

##### 3.获取图片常用属性

- img.shape：打印图片的高、宽和通道数（当图片为灰度图像时，颜色通道数为1，不显示）
- img.size：打印图片的像素数目
- img.dtype：打印图片的格式

##### 4.图片颜色通道的分离和合并

- cv2.split(m)：将图片分离为三个颜色通道
- cv2.merge(mv)：将各个通道合并为一张图片

##### 5.两张图片相加，改变对比度和亮度

- cv2.addWeighted(src1, alpha, src2, w2，beta)：带权相加，alpha的值改变对比度，beta控制亮度

##### 6.像素运算

- cv2.add(m1, m2)：对应像素相加

- cv2.subtract(m1, m2)：对应像素相减

- cv2.multiply(m1, m2)：对应像素相乘

- cv2.divide(m1, m2)：对应像素相除
- cv2.mean(img)：像素均值
- cv2.meanStdDev(img)：像素均值和标准差
- cv2.bitwise_and(m1, m2)：像素与运算
- cv2.bitwise_or(m1, m2)：像素或运算
- cv2.bitwise_not(m1, m2)：像素非运算
- cv2.bitwise_xor(m1, m2) ：像素异或运算

##### 7.色彩转换

- cv2.cvtColor(src, code, dst, dstCn)

##### 8.图像模糊（图像平滑）

- cv2.blur(src, ksize, dst, anchor, borderType) ：图像均值平滑
- cv2.medianBlur(src, ksize, dst)：图像中值平滑
- cv2.GaussianBlur(src, ksize, sigmaX, dst, sigmaY, borderType)：图像高斯平滑
- cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace, dst, borderType)：图像双边模糊

##### 9.二值化

- cv2.threshold(src, thresh, maxval, type, dst)：将图像的每个像素点按阈值进行二值化

- cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C, dst)：将图像的每个像素点按自适应阈值进行二值化

##### 10.图像直方图

- cv2.calcHist(images, channels, mask, histSize, ranges, hist, accumulate)：直方图统计图像内各个灰度级出现的次数，横坐标是图像中各像素点的灰度级，纵坐标是具有该灰度级（像素值）的像素个数

##### 11.模板匹配

- cv2.matchTemplate(image, templ, method, result, mask)：以模板图像为滑动窗口，在原图中查找各个位置的相似度
- cv2.minMaxLoc(src, mask)：返回矩阵中的最小值和最大值，以及它们的位置

##### 12.图像金字塔（上采样和下采样）

- cv2.pyrDown(src, dst, dstsize, borderType)：下采样
- cv2.pyrUp(src, dst, dstsize, borderType)：上采样
- cv2.resize(src, dsize, dst, fx, fy, interpolation)：将图像调整到指定大小

##### 13.图像边缘检测

- cv2.Sobel(src, ddepth, dx, dy, dst, ksize, scale, delta, borderType) ：一阶导
- cv2.Scharr(src, ddepth, dx, dy, dst, scale, delta, borderType)：一阶导
- cv2.Laplacian(src, ddepth, dst, ksize, scale, delta, borderType)：二阶导
- cv2.Canny(image, threshold1, threshold2, edges, apertureSize, L2gradient)：高斯滤波+非极大值抑制+双阈值，来确定图像边缘

##### 14.图像的仿射变换

- cv2.warpAffine(src, M, dsize, dst, flags, borderMode, borderValue)：仿射变换，改变图像中

- cv2.getRotationMatrix2D(center, angle, scale)：获取旋转后的图像仿射矩阵

##### 15.其他

cv2.xfeatures2d.BriefDescriptorExtractor_create() # Brief描述符提取器
cv2.SIFT_create() # 使用比例不变特征变换（SIFT）算法提取关键点和计算描述符
cv2.ORB_create() # 面向Brief的关键点检测器和描述符提取器
cv2.GFTTDetector_create() # 创建特征检测器，查找图像中最突出的角点
cv2.BFMatcher_create() # 创建蛮力匹配器
cv2.FlannBasedMatcher_create() # 基于Flann的描述符匹配器
cv.estimateAffinePartial2d() # 计算两个 2D 点集之间具有 4 个自由度的最优有限仿射变换
cv.findHomography() # 生成两个平面的透视变换矩阵
cv.Rodrigues() # 将旋转矩阵转化为旋转向量，反之亦然
cv.getPerspectiveTransform(src, dst, solveMethod) # 从四对对应的点计算一个透视变换矩阵
cv.warpPerspective() # 将透视转换应用于图像
cv.warpAffine() # 仿射变换
cv.invert() # 求矩阵的逆或伪逆矩阵
cv.transfome() # 执行每个数组元素的矩阵转换
cv.getRectSubPix() # 从原图中截取矩形图像
cv.intersectConvexConvx() # 检测两个凸多边形的交点
cv.contourArea() # 计算轮廓面积
cv.imencode() # 将图片转成buffer
cv.imdecode() # 将buffer转成图片
cv.fillConvexPoly() # 填充凸多边形
cv.getRotationMatrix2D() # 获取旋转后的图像仿射矩阵
cv.warpAffine() # 仿射变换，改变图像中
cv.boxPoints() # 查找旋转矩形的四个顶点。 用于绘制旋转的矩形
cv.fillPoly() # 填充由一个或多个多边形包围的区域。
cv.findNonZero() # 返回非零像素的位置列表
cv.minAreaRect() # 查找包含输入 2D 点集的最小区域的旋转矩形
cv.polylines() # 绘制多条多边形曲线