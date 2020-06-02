# -*- coding: utf-8 -*-
from __future__ import division
import cv2
import numpy as np
import math

# 读取鱼眼图片
img = cv2.imread("images/img10.jpg")
# 设置灰度阈值
T = 40

# 转换为灰度图片
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 提取原图大小
rows, cols = img.shape[:2]
print(rows, cols)


# 从上向下扫描
for i in range(0, rows, 1):
    for j in range(0, cols, 1):
        if img_gray[i, j] >= T:
            if img_gray[i + 1, j] >= T:
                top = i
                break
    else:
        continue
    break
print('top =', top)


# 从下向上扫描
for i in range(rows - 1, -1, -1):
    for j in range(0, cols, 1):
        if img_gray[i, j] >= T:
            if img_gray[i - 1, j] >= T:
                bottom = i
                break
    else:
        continue
    break
print('bottom =', bottom)


# 从左向右扫描
for j in range(0, cols, 1):
    for i in range(top, bottom, 1):
        if img_gray[i, j] >= T:
            if img_gray[i, j + 1] >= T:
                left = j
                break
    else:
        continue
    break
print('left =', left)


# 从右向左扫描
for j in range(cols - 1, -1, -1):
    for i in range(top, bottom, 1):
        if img_gray[i, j] >= T:
            if img_gray[i, j - 1] >= T:
                right = j
                break
    else:
        continue
    break
print('right =', right)


# 计算有效区域半径
R = max((bottom - top) / 2, (right - left) / 2)
print('R =', R)


# 提取有效区域
img_valid = img[top:int(top + 2 * R+1), left:int(left + 2 * R+1)]   #注意+1
cv2.imwrite('./TestResults/result.jpg', img_valid)


#经度矫正法
m, n, k = img_valid.shape[:3]
print('m,n,k',m,n,k)
result = np.zeros((m,n,k))

for i in range(m):
    for j in range(n):
        u = j - R     #修改后可以的
        v = R - i
        # u = (2*i) /(m-1)   #原论文中的公式有误 黑的
        # v = (2*j) /(n-1)
        r = math.sqrt(u * u + v * v)

        if(r == 0):
            fi = 0
        elif (u >= 0):
            fi = math.asin(v/r)
        else:
            fi = math.pi - math.asin(v/r)

        f = R * 2 / math.pi
        theta = r / f

        # f = 1
        # theta = r * math.pi /2
        # f = R * 2 / math.pi
        # theta = r/f
        # theta = (r / R) * math.pi / 2

        x=f * math.sin(theta) * math.cos(fi)
        y=f * math.sin(theta) * math.sin(fi)
        z=f * math.cos(theta)

        rr= math.sqrt(x * x + z * z)
        sita = math.pi / 2 - math.atan( y /rr)
        if(z>=0):
            fai = math.acos(x/rr)
        else:
            fai= math.pi - math.acos(x/rr)

        xx = round(f * sita)
        yy = round(f * fai)
        # xx = round(f * sita)
        # yy = round(2 * R - f * fai)

        if ((xx < 1) | (yy < 1) | (xx > m) | (yy > n)):
            continue

        result[xx,yy,0] = img_valid[i, j, 0]
        result[xx,yy,1] = img_valid[i, j, 1]
        result[xx,yy,2] = img_valid[i, j, 2]
        # print("内循环",j)
    print("外循环", i)

Undistortion = np.uint8(result)
cv2.imwrite('./TestResults/Undistortion111.jpg', Undistortion)
a, b, c = Undistortion.shape[:3]
print('a,b,c',a,b,c)

# 显示图片
cv2.namedWindow("yuantu", 0)
cv2.resizeWindow("yuantu", 640, 480)
cv2.imshow("yuantu", img)

cv2.namedWindow("result", 0)
cv2.resizeWindow("result", 640, 480)
cv2.imshow("result", Undistortion)
cv2.waitKey(0)
