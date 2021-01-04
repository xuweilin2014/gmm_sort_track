import cv2 as cv

box = cv.imread("../images/big_pot.jpg")
box_in_sence = cv.imread("../images/leaves.jpg")
cv.imshow("box", box)
cv.imshow("box_in_sence", box_in_sence)

# 创建 SIFT 特征检测器
sift = cv.xfeatures2d.SIFT_create()

# 特征点提取与描述子生成
kp1, des1 = sift.detectAndCompute(box, None)
kp2, des2 = sift.detectAndCompute(box_in_sence, None)

# 暴力匹配
bf = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE)
matches = bf.match(des1, des2)

# 绘制最佳匹配
matches = sorted(matches, key=lambda x: x.distance)
result = cv.drawMatches(box, kp1, box_in_sence, kp2, matches[:5], None)
cv.imshow("-match", result)
cv.imwrite("../images/result.jpg", result)
cv.waitKey(0)
cv.destroyAllWindows()
