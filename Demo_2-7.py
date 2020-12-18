import cv2
from matplotlib.pyplot import phase_spectrum, switch_backend
import numpy as np
import math
import matplotlib.pyplot as plt

##### 시작 #####

## 1. Image 출력
img = cv2.imread('../img/1.jpg')
resize_img = cv2.resize(img, (954, 954))
grabCut = resize_img.copy()

## 2. onMouse Func
mouse_pressed = False
x = y = w = h = 0

# 2.1 관심영역 설정
def onMouse(event, _x, _y, flags, param):
    global grabCut, x, y, w, h, mouse_pressed

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed = True
        x, y = _x, _y
        grabCut = resize_img.copy()

    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_pressed:
            grabCut = resize_img.copy()
            cv2.rectangle(grabCut, (x, y), (_x, _y), (0, 255, 0), 1)

    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed = False
        w, h = _x - x, _y - y

cv2.namedWindow('image')
cv2.setMouseCallback('image', onMouse)

# 2.2 GrabCut 함수 시작
while True:
    cv2.imshow('image', grabCut)
    k = cv2.waitKey(1)

    if k ==ord('c') and not mouse_pressed:
        if w * h > 0:
            break
cv2.destroyAllWindows()

gMask1 = np.zeros(resize_img.shape[:2], np.uint8) # Grabcut 마스크를 씌울 이미지
gMask1, bgdModel, fgdModel = cv2.grabCut(resize_img, gMask1, (x, y, w, h),
                                         None, None, 5, cv2.GC_INIT_WITH_RECT) # Grabcut 함수
grabCut = resize_img.copy()
grabCut[(gMask1 == cv2.GC_PR_BGD) | (gMask1 == cv2.GC_BGD)] //= 3 # 마스크를 씌운 이미지 생성
cv2.imshow('image', grabCut)
cv2.waitKey()
cv2.destroyAllWindows()

# 2.3 GrabCut 오류 수정
gMask2 = cv2.GC_BGD
gMask_cntr = {cv2.GC_BGD: (0, 0, 0),
              cv2.GC_FGD: (255, 255, 255)} # BGD: 배경마스크를 검정으로 표시, FGD: 전경 마스크를 흰색으로 표시

def onMouse(event, x, y, flags, param): # 수정함수
    global mouse_pressed

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed = True
        cv2.circle(gMask1, (x, y), 3, gMask2, cv2.FILLED)
        cv2.circle(grabCut, (x, y), 3, gMask_cntr[gMask2], cv2.FILLED)

    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_pressed:
            cv2.circle(gMask1, (x, y), 3, gMask2, cv2.FILLED)
            cv2.circle(grabCut, (x, y), 3, gMask_cntr[gMask2], cv2.FILLED)

    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed = False

cv2.namedWindow('image')
cv2.setMouseCallback('image', onMouse)

while True:
    cv2.imshow('image', grabCut)
    k = cv2.waitKey(1)

    if k == ord('c') and not mouse_pressed:
        break

    elif k == ord('f'): # f 버튼으로 전경 및 배배 설정을 변경하고 마우스 클릭으로 전경 및 배경 설정
        gMask2 = cv2.GC_FGD - gMask2

cv2.destroyAllWindows()

gMask1, bgdModel, fgdModel = cv2.grabCut(resize_img, gMask1, None,
                                         bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
grabCut= resize_img.copy()
grabCut[(gMask1 == cv2.GC_PR_BGD) | (gMask1 == cv2.GC_BGD)] = 0
stem = grabCut.copy()

## 3. GrabCut 완료된 이미지를 활용한 AR 객체 추출
# 3.1 Mask 생성
red1 = np.array([0,50,50])
red2 = np.array([20,255,255])
red3 = np.array([160,50,50])
red4 = np.array([180,255,255])

yellow1 = np.array([25,50,50])
yellow2 = np.array([30,255,255])

blue1 = np.array([90,50,50])
blue2 = np.array([120,255,255])

# 3.2 이미지 변화
grabCut[:, :, 0] = cv2.equalizeHist(grabCut[:, :, 0]) # 히스토그램 평활화 (밝기 변화)
pyrMean = cv2.pyrMeanShiftFiltering(grabCut, 30, 30, None, 2) # 평균 이동 필터
hsv_img = cv2.cvtColor(pyrMean, cv2.COLOR_BGR2HSV) # HSV 색상공간 변화

# 3.3 색상추출
mask_red1 = cv2.inRange(hsv_img, red1, red2)
mask_red2 = cv2.inRange(hsv_img, red3, red4)
mask_yellow = cv2.inRange(hsv_img, yellow1, yellow2)
mask_blue = cv2.inRange(hsv_img, blue1, blue2)

# 3.4 객체추출
red_obj1 = cv2.bitwise_and(pyrMean, pyrMean, mask=mask_red1)
red_obj2 = cv2.bitwise_and(pyrMean, pyrMean, mask=mask_red2)
red_obj = cv2.bitwise_or(red_obj1, red_obj2)
yellow_obj = cv2.bitwise_and(pyrMean, pyrMean, mask=mask_yellow)
blue_obj = cv2.bitwise_and(pyrMean, pyrMean, mask=mask_blue)

# 3.5 AR 객체 이진화
r_gray = cv2.cvtColor(red_obj, cv2.COLOR_BGR2GRAY) # 회색조 영상
y_gray = cv2.cvtColor(yellow_obj, cv2.COLOR_BGR2GRAY)

r_adapthr = cv2.adaptiveThreshold(r_gray, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 2) # 적응형 이진화
y_adapthr = cv2.adaptiveThreshold(y_gray, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 2)

r_kerl1 = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6)) # 모폴로지 연산을 위한 사각 마스크(6x6) 생성
r_kerl2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
y_kerl = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))

r_open = cv2.morphologyEx(r_adapthr, cv2.MORPH_OPEN, r_kerl1) # r_kerl1을 이용한 모폴로지 오픈 연산 적용
y_open = cv2.morphologyEx(y_adapthr, cv2.MORPH_OPEN, y_kerl)

r_close = cv2.morphologyEx(r_open, cv2.MORPH_CLOSE, r_kerl2) # r_kerl2을 이용한 모폴로지 닫기 연산 적용
y_close = cv2.morphologyEx(y_open, cv2.MORPH_CLOSE, y_kerl)

ret1, r_thr = cv2.threshold(r_close, 127, 255, cv2.THRESH_BINARY_INV) # 이진화 이미지 반전(컨투어 생성을 위해)
ret2, y_thr = cv2.threshold(y_close, 127, 255, cv2.THRESH_BINARY_INV)

# 3.6 컨투어 생성
r_contour, rhr = cv2.findContours(r_thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
y_contour, yhr = cv2.findContours(y_thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

r_contr = r_contour[0] # contour index 정의
y_contr = y_contour[0]

# rx, ry, rw, rh = cv2.boundingRect(r_contr)
# yx, yy, yw, yh = cv2.boundingRect(y_contr)
# cv2.rectangle(red_obj, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 2)
# cv2.rectangle(yellow_obj, (yx, yy), (yx + yw, yy + yh), (30, 255, 255), 2)

r_rect = cv2.minAreaRect(r_contr) # 컨투어를 감싸는 사각형 생성
y_rect = cv2.minAreaRect(y_contr)

r_box = cv2.boxPoints(r_rect) # 사각형 꼭지점 좌표생성
y_box = cv2.boxPoints(y_rect)
r_box = np.int0(r_box) # 꼭지점 좌표 정수화
y_box = np.int0(y_box)

top_x = r_box[2][0] # r_rect 꼭지점 좌표 생성
top_y = r_box[2][1]
top_x1 = r_box[3][0]
top_y1 = r_box[3][1]
bott_x = r_box[0][0]
bott_y = r_box[0][1]
bott_x1 = r_box[1][0]
bott_y1 = r_box[1][1]

cv2.drawContours(red_obj, [r_box], -1, (0, 0, 255), 3) # 컨투어 그리기
cv2.drawContours(yellow_obj, [y_box], -1, (30, 255, 255), 3)

# 3.7 흉고높이 및 흉고직경 픽셀 정의
r_mmt = cv2.moments(r_contr) # 중심좌표 계산을 위한 모멘트 생성
y_mmt = cv2.moments(y_contr)

r_cx = int(r_mmt['m10'] / r_mmt['m00']) # 중심좌표 계산
r_cy = int(r_mmt['m01'] / r_mmt['m00'])
y_cx = int(y_mmt['m10'] / y_mmt['m00'])
y_cy = int(y_mmt['m01'] / y_mmt['m00'])

cv2.circle(red_obj, (r_cx, r_cy), 3, (0, 255, 0), -1)
cv2.circle(yellow_obj, (y_cx, y_cy), 3, (0, 255, 0), -1)

bh_line = cv2.line(red_obj, (r_cx, r_cy), (y_cx, y_cy),
                   (0, 255, 0), 3, cv2.LINE_AA) # 흉고높이를 표시하는 직선 그리기
# dbh_line = cv2.line(red_obj, (x, y), (x+w, y+h),
#                     (0, 255, 0), 3, cv2.LINE_AA) # 흉고직경을 표시하는 직선 그리기

a = abs(r_cx - y_cx)
b = abs(r_cy - y_cy)
breast_height = int(math.sqrt((a**2) + (b**2))) # 흉고높이
diameter = abs(top_x - top_x1) # 흉고직경
print('Breast Height: ', breast_height, '\nDiameter1: ', diameter)

## 4. 수간추출
final_pyr = cv2.pyrMeanShiftFiltering(stem, 30, 30, None, 2)
stem_gray = cv2.cvtColor(final_pyr, cv2.COLOR_BGR2GRAY)
stem_thr = cv2.adaptiveThreshold(stem_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 5)
ret, stem_thr = cv2.threshold(stem_thr, 127, 255, cv2.THRESH_BINARY_INV)

stem_kerl = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
close_img = cv2.morphologyEx(stem_thr, cv2.MORPH_CLOSE, stem_kerl)

stem_contour, hr = cv2.findContours(close_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
stm_contr = stem_contour[0]
final_pyr1= final_pyr.copy()

cv2.drawContours(final_pyr, [stm_contr], -1, (0, 255, 0), 3)

# for i in stem_contour: # contour 마다 파란색 원 그리기
#     for j in i:
#         cv2.circle(final_pyr, tuple(j[0]), 1, (255, 0, 0), -1)

# 4.1 컨투어 좌표점을 이용한 면적 계산용 네모 박스 or 등고선 생성  <첫번째 좌표점 -> contours[0][0][9][0](x) contours[0][0][0][1](y)>
                                                        # <두번째 좌표점 -> contours[0][1][0][0](x) contours[0][1][0][1](y)>
start_x = 0 # 화면 좌표 정의
end_x = 954

y_minIndex = np.argmin(stem_contour[0], axis=0)
y_min = stem_contour[0][y_minIndex[0][1]][0][1]

line_img = np.zeros((954, 954, 3), np.uint8) # 954 x 954 x 3인 검정색 화면 생성

for i in range(0, 954, 10): # 화면 전체에 30 간격으로 라인 생성 ***여기서 30은 0.5cm로 환산한 것이 아님!!! 임의로 값을 설정한 거임***
    if y_cy - i <= y_min:
        break
    cv2.line(line_img, (start_x, y_cy - i), (end_x, y_cy - i), (0, 255, 255), 1, cv2.LINE_AA)

stm_mmt = cv2.moments(stm_contr)

stm_cx = int(stm_mmt['m10'] / stm_mmt['m00'])
stm_cy = int(stm_mmt['m01'] / stm_mmt['m00'])

rows, cols = final_pyr.shape[:2]
fill_mask = np.zeros((rows+2, cols+2), np.uint8)
cv2.floodFill(final_pyr, fill_mask, (stm_cx, stm_cy), (255, 255, 255), (10, 10, 10), (10, 10 ,10)) #컨투어 내부를 같은 색상으로 채우기
cv2.floodFill(final_pyr, fill_mask, (r_cx, r_cy), (255, 255, 255), (10, 10, 10), (10, 10 ,10))
cv2.floodFill(final_pyr, fill_mask, (y_cx, y_cy), (255, 255, 255), (10, 10, 10), (10, 10 ,10))

new = cv2.bitwise_and(line_img, final_pyr)
new_gray = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
ret3, new_thr = cv2.threshold(new_gray, 10, 255, cv2.THRESH_BINARY)
new_thr1 = cv2.adaptiveThreshold(new_gray, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 2)
ret4, new_thr2 = cv2.threshold(new_thr1, 127, 255, cv2.THRESH_BINARY_INV)

kerl = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
new_morph1 = cv2.morphologyEx(new_thr, cv2.MORPH_CLOSE, kerl)
new_morph2 = cv2.morphologyEx(new_thr2, cv2.MORPH_CLOSE, kerl)

stm_contour1, hr1 = cv2.findContours(new_morph1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
stm_contour2, hr2 = cv2.findContours(new_morph2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

final = final_pyr.copy()

c_list = []
stm_Dlist = []
stem_Volume = 0
box_id = 1;
left_list = []
right_list = []
#test용
cx_list = []
cy_list = []


for i in stm_contour1: # 수간 굽음도를 구하기 위한 중심점 출력
    contr1 = i[0]
    line_rect1 = cv2.minAreaRect(i)
    box1 = cv2.boxPoints(line_rect1)
    box1 = np.int0(box1)
    cv2.drawContours(final_pyr, [i], -1, (0, 0, 255), 1)

    # x, y, w, h = cv2.boundingRect(i)
    # cv2.rectangle(final_pyr, (x, y), (x+w, y+h), (0, 0, 255), 2)

    moment = cv2.moments(i)
    cx = int(moment['m10'] / moment['m00'])
    cy = int(moment['m01'] / moment['m00'])
    cv2.circle(final_pyr, (cx, cy), 3, (0, 255, 0), -1)
    cv2.putText(final_pyr, str(box_id), (cx+60, cy+5), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (255, 255, 0), 1)
    box_id += 1
    c_list.append((cx, cy))

    x1 = box1[0][0]
    x2 = box1[2][0]
    y1 = box1[0][1]
    y2 = box1[2][1]
    x3 = box1[1][0]
    x4 = box1[3][0]
    y3 = box1[1][1]
    y4 = box1[3][1]

    cv2.line(final_pyr, (x3, y3), (x4, y4), (255, 0, 0), 2)

    cv2.circle(final_pyr, (x1, y1), 2, (255, 0, 0), -1) #파 => 오른쪽 밑 [0][0] [0][1] 왼쪽 밑
    cv2.circle(final_pyr, (x2, y2), 2, (30, 255, 255), -1) #노 => 왼쪽 위 [2][0] [2][1] 오른쪽 위
    cv2.circle(final_pyr, (x3, y3), 2, (30, 200, 0), -1) #초 => 왼쪽 밑 [1][0] [1][1] 왼쪽 위
    cv2.circle(final_pyr, (x4, y4), 2, (0, 0, 255), -1) #빨 => 오른쪽 위 [3][0] [3][1] 오른쪽 밑

    left_list.append((x1,y1))
    left_list.append((x3,y3))
    right_list.append((x2,y2))
    right_list.append((x4,y4))
    cx_list.append(cx)
    cy_list.append(cy)

    stm_dbh = abs(x3 - x4)
    stm_Dlist.append(stm_dbh)
    print('stm_dbh: ', stm_dbh)
    print('list: ', stm_Dlist) # 각 stm_dbh 값의 list


# for i in stm_contour2:
#     contr2 = i[0]
#     line_rect2 = cv2.minAreaRect(i)
#     box2 = cv2.boxPoints(line_rect2)
#     box2 = np.int0(box2)
#     cv2.drawContours(final, [i], -1, (0, 0, 255), 1)
#     moment = cv2.moments(i)
#     cx = int(moment['m10'] / moment['m00'])
#     cy = int(moment['m01'] / moment['m00'])
#     cv2.circle(final, (cx, cy), 3, (0, 255, 0), -1)
#     cv2.putText(final, str(box_id), (cx + 60, cy + 5), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (255, 255, 0), 1)
#     box_id += 1
#     c_list.append((cx, cy))
#
#     x1 = box2[0][0]
#     x2 = box2[2][0]
#     y1 = box2[0][1]
#     y2 = box2[2][1]
#     x3 = box2[1][0]
#     x4 = box2[3][0]
#     y3 = box2[1][1]
#     y4 = box2[3][1]
#
#     cv2.line(final, (x3, y3), (x4, y4), (255, 0, 0), 2)
#
#     cv2.circle(final, (x1, y1), 2, (255, 0, 0), -1)  # 파 => 오른쪽 밑 [0][0] [0][1] 왼쪽 밑
#     cv2.circle(final, (x2, y2), 2, (30, 255, 255), -1)  # 노 => 왼쪽 위 [2][0] [2][1] 오른쪽 위
#     cv2.circle(final, (x3, y3), 2, (30, 200, 0), -1)  # 초 => 왼쪽 밑 [1][0] [1][1] 왼쪽 위
#     cv2.circle(final, (x4, y4), 2, (0, 0, 255), -1)  # 빨 => 오른쪽 위 [3][0] [3][1] 오른쪽 밑
#
#     stm_dbh = abs(x3 - x4)
#     print('stm_dbh2: ', stm_dbh)
#

for i, val in enumerate(c_list): # 중심을 잇는 직선 그리기
    if val == c_list[-1]:
        break
    cv2.line(final_pyr, val, c_list[i + 1], (110, 20, 20), 1, cv2.LINE_AA)
    cv2.line(final, val, c_list[i + 1], (110, 20, 20), 1, cv2.LINE_AA)

# 4.2 단재적 및 이용재적 산출
for i in range(0, len(stm_Dlist) - 1): # 구간별 Smalian식을 이용하여 재적구하기
    stem_Volume += round(((((stm_Dlist[i]**2) * math.pi) + ((stm_Dlist[i+1]**2) * math.pi)) * 0.5) / 2, 4)

    print('stem_Volume: ', stem_Volume)

print('\n\n\n\n\n ======= testing =====') 

	# 수간고별 직사각형 그리기 
	# 공제량 부분인지 아닌지 판단 
	# 공제량 면적 계산(다각형 면적 구하기) 
	# 직사각형 면적구하고 
	# 직사각형 - 다각형 면적 = 실제 사용가능한 목재 면적
	# 실제 사용가능한 목재 면적에 이용재적 공식 대입
	# 각 계산식 별 결과 출력
	# 가장 많은 이용재적 산출하는 수간(라벨 번호) 결과 출력


result_list1 = []

def masks(vec): # 참고 : https://stackoverrun.com/ko/q/12969470
    d= np.diff(vec)
    dd = np.diff(d)

    # Mask of locations where graph goes to vertical or horizontal, depending on vec
    to_mask = ((d[:-1] != 0) & (d[:-1] == -dd))
    # Mask of locations where graph comes from vertical or horizontal, depending on vec
    from_mask = ((d[1:] != 0) & (d[1:] == dd))
    return to_mask, from_mask

to_vert_mask, from_vert_mask = masks(cx_list)
to_horiz_mask, from_horiz_mask = masks(cy_list)

def apply_mask(mask, x, y):
       return x[1:-1][mask], y[1:-1][mask]

to_vert_x, to_vert_y = apply_mask(to_vert_mask, np.array(cx_list), np.array(cy_list))
from_vert_x, from_vert_y = apply_mask(from_vert_mask, np.array(cx_list), np.array(cy_list))
to_horiz_x, to_horiz_y = apply_mask(to_horiz_mask, np.array(cx_list), np.array(cy_list))
from_horiz_x, from_horiz_y = apply_mask(from_horiz_mask, np.array(cx_list), np.array(cy_list))

lenChk = False
#clist - 컨투어 중심점 (x,y) 리스트, left_list - 왼쪽 좌표 리스트, right_list = 오른쪽 좌표 리스트
if(len(left_list) == len(right_list)):
    final_index = len(left_list)
    lenChk = True
    plt.subplot(1, 2, 1)
    plt.plot(cx_list,cy_list, 'b-')
    plt.plot(to_vert_x, to_vert_y, 'r^', label='Plot goes vertical')
    plt.plot(from_vert_x, from_vert_y, 'kv', label='Plot stops being vertical')
    plt.plot(to_horiz_x, to_horiz_y, 'r>', label='Plot goes horizontal')
    plt.plot(from_horiz_x, from_horiz_y, 'k<', label='Plot stops being horizontal')
    plt.legend()
    # plt.subplot(1, 2, 1)
    # plt.plot(cx_list,cy_list, marker='o')
    # # plt.plot(cx_list,cy_list, marker='o')
    # plt.xlabel('center x reverse val')
    # plt.ylabel('center y val')
    # plt.legend(['x-reverse', 'y'])
    plt.gca().invert_yaxis() #y축 범위 역순 조정
    plt.subplot(1, 2, 2)
    # plt.title('first Subplot')
    # plt.plot(cy_list,cx_list[::-1], marker='o')
    # plt.xlabel('center y val')
    # plt.ylabel('center x reverse val')
    # plt.legend(['y', 'x'])
    # plt.title('Second Subplot')
    to_horiz_x, to_horiz_y = apply_mask(to_horiz_mask, np.array(cx_list), np.array(cy_list))
    to_horiz_x, to_horiz_y = to_horiz_x[:-1], to_horiz_y[:-1]

    plt.plot(cx_list,cy_list, 'b-')
    plt.plot(*apply_mask(to_vert_mask, np.array(cx_list), np.array(cy_list)), 'r^', label='Plot goes vertical')
    plt.plot(*apply_mask(from_vert_mask, np.array(cx_list), np.array(cy_list)), 'kv', label='Plot stops being vertical')
    plt.plot(to_horiz_x, to_horiz_y, 'r>', label='Plot goes horizontal')
    plt.legend()
    plt.gca().invert_yaxis() #y축 범위 역순 조정

    result_list1 = np.append(result_list1,apply_mask(to_vert_mask, np.array(cx_list), np.array(cy_list)))
    result_list1 = np.append(result_list1,apply_mask(from_vert_mask, np.array(cx_list), np.array(cy_list)))
    result_list1 = np.append(result_list1, (to_horiz_x, to_horiz_y))
    result_list1 = np.unique([result_list1]) # 중복 제거
    print('\n===\n변곡점 list (result_list1) : ',result_list1,' , 갯수:',len(result_list1),'\n===\n')
    # r_idx_list1 = np.where(np.array(cy_list) == result_list1[0])
    r_idx_list1= []
    for i in range(0,len(result_list1)):
        idx = np.where(cy_list == result_list1[i])
        # print(idx[0])
        if(idx[0].size != 0): #numpy array size 체크
            # print(idx[0][0])
            r_idx_list1.append(idx[0][0])
    print('\n===\n변곡점 인덱스 list (r_idx_list1) : ',r_idx_list1,' , 갯수:',len(r_idx_list1),'\n===\n')

#lx_min_idx = np.min(left_list[:][0]) #값만 꺼낼 때는 np.min / np.max

# print(left_list[:], type(left_list), type(left_list[0,:]))
#print('lx_min_idx',lx_min_idx)
# lx_max_idx = np.argmax(left_list[:,0])
# rx_min_idx = np.argmin(right_list[:,0])
# rx_max_idx = np.argmax(right_list[:,0])

def test1(size):
    # 공제량 부분인지 아닌지 판단                      # 485 ~ 495
    # 수간고별 직사각형 그리기                         # 497 ~ 505
	# 직사각형 면적구하기                              # 497 ~ 505 
    # 공제량 면적 계산(다각형 면적 구하기)              # 511 ~ 547
	# 직사각형 - 다각형 면적 = 실제 사용가능한 목재 면적 # 511~ 547 , 현재 각 부분마다 바로 제거하는 형태로 구현
	# 실제 사용가능한 목재 면적에 이용재적 공식 대입            # 573 ~
	# 각 계산식 별 결과 출력                                  # ''
	# 가장 많은 이용재적 산출하는 수간(라벨 번호) 결과 출력     # ''
    val_list = []
    useValSum = None
    min_width = 10 # 최소 목재 width 사이즈 지정,    (20.11.25) !!실제 12cm 값으로 수정하기!!
    #min_height = 10 # 최소 목재 height 사이즈 지정, (20.11.25) !!실제 1.8m값으로 수정하기!!

    for i in range(1,size):
        #area_tmp = w_tmp*h_tmp #직사각형 면적(math.PI를 곱해서 원기둥으로 구할지?)
        #공제량 부분인지 판단 (굽음이 있는 경우)
        # -----
        if i in r_idx_list1 :
            #맞을 경우, 공제량 면적 계산
            # ----
            #방법 1. (조건문 체크 -> 직사각형 면적 계산 -> 나머지 좌표들로 다각형 면적 계산)
            a1 = left_list[i]; b1 = right_list[i] # 상단 좌표
            c1 = left_list[i-1] ; d1 = right_list[i-1] # 하단 좌표
            
            h_val = b1[1] - d1[1] # 오른쪽 좌표 기준으로 계산

            #조건문 체크
            if a1[0] - c1[0] < 0 or c1[0] - a1[0] < 0 or b1[0] - d1[0] < 0 or d1[0] - b1[0] < 0:
                #직사각형 면적부터 계산
                if b1[0] - a1[0] >= min_width:
                    w_val = b1[0] - a1[0]
                    # h_val = b1[1] - d1[1] # 오른쪽 좌표 기준으로 계산
                    first_rect = w_val*h_val
                #b1[0] - a1[0]이 지정해둔 최소 width 사이즈보다 작으면 직사각형 가로 크기는 지정 크기로 할 것
                else:
                    w_val = min_width
                    # h_val = b1[1] - d1[1] # 오른쪽 좌표 기준으로 계산
                    first_rect = w_val*h_val

                #나머지 좌표들로 다각형 면적 계산 
                #직사각형 내에 나머지 부분이 있는지부터 확인 필요
                
                # 왼쪽 체크
                if a1[0] - c1[0] < 0: # \ 상태
                    l = [(a1[0],c1[1]), a1, c1]
                    # N각형인지
                    # if N각형일 경우:
                        # l.append(좌표) #좌표 추가 
                    arr = np.array(l)
                    left_tmp = cv2.contourArea(arr)
                    first_rect = first_rect - left_tmp
                    pass
                if c1[0] - a1[0] < 0: # / 상태
                    l = [c1, (c1[0], a1[1]), a1]
                    # N각형인지
                    # if N각형일 경우:
                        # l.append(좌표) #좌표 추가 
                    arr = np.array(l)
                    left_tmp = cv2.contourArea(arr)
                    first_rect = first_rect - left_tmp
                    pass
                #오른쪽 체크
                if b1[0] - d1[0] < 0 : # \ 상태
                    l = [b1, d1, ()]
                    # N각형인지
                    # if N각형일 경우:
                        # l.append(좌표) #좌표 추가 
                    arr = np.array(l)
                    right_tmp = cv2.contourArea(arr)
                    first_rect = first_rect - right_tmp
                    pass
                if d1[0] - b1[0] < 0: # / 상태
                    l = [()]
                    # N각형인지
                    # if N각형일 경우:
                        # l.append(좌표) #좌표 추가 
                    arr = np.array(l)
                    right_tmp = cv2.contourArea(arr)
                    first_rect = first_rect - right_tmp
                    pass

            else: #변곡점임에도 불구하고, 통 직사각형인 경우 or 생각못한 예외 둘 중 하나
                if b1[0] - a1[0] >= min_width:
                    w_val = b1[0] - a1[0]
                    first_rect = w_val*h_val
            
                else:
                    w_val = min_width
                    first_rect = w_val*h_val
                print('통 직사각형 or 예외')

            val_list.append(first_rect)
            pass
        # -----
        else:
            #공제량 부분이 아닌경우, (굽음이 없는 경우)
            
            a1 = left_list[i]; b1 = right_list[i] # 상단 좌표
            c1 = left_list[i-1] ; d1 = right_list[i-1] # 하단 좌표
            w_val = b1[0] - a1[0]
            h_val =  b1[1] - d1[1]
            first_rect = w_val*h_val
            val_list.append(first_rect)

            pass
    # 실제 사용가능한 목재 면적에 이용재적 공식 대입
    # 각 계산식 별 결과 출력
	# 가장 많은 이용재적 산출하는 수간(라벨 번호) 결과 출력
    pass


def test2(size):
    val_list = []
    useValSum = None
    min_width = 10 # 최소 목재 width 사이즈 지정, 현재 10으로 설정
    #min_height = 10 # 최소 목재 height 사이즈 지정, 필요할까봐 일단 Keep함

    for i in range(1,size):
        #area_tmp = w_tmp*h_tmp #직사각형 면적(math.PI를 곱해서 원기둥으로 구할지?)
        #공제량 부분인지 판단 (굽음이 있는 경우)
        # -----
        if i in r_idx_list1 :
            #맞을 경우, 공제량 면적 계산
            #방법 2. (cv2로 직사각형 이미지 그리기 -> 부분 반전 -> 공제량 부분 좌표 추출 -> 다각형 면적 계산 -> 직 - 다)
            # ----
            pass
        # -----
        else:
            #공제량 부분이 아닌경우, (굽음이 없는 경우)
            #이용재적 결과 값에 더해둔다.
            #useValSum += area_tmp
            pass
    # 실제 사용가능한 목재 면적에 이용재적 공식 대입
    # 각 계산식 별 결과 출력
	# 가장 많은 이용재적 산출하는 수간(라벨 번호) 결과 출력
    pass

print()


# Smalian Eq. => ((math.pi * (d1**2) + math.pi * (d2**2)) * length)) / 2

# 결함 찾기 (측부결함, 손상결함) : 1. 미국 임야청에서 사용하는 공제 공식 => D = (W * T * L) / 15 { W: 결함 부분의 너비, T: 결함 부분의 두께, L: 결함 부분의 길이 } 산림측정학 p.31 ~ 35

# https://oakmissouri.org/forestfunction/ 참고 


cv2.imshow('contour_None1', final_pyr)
if(lenChk):
    #test1(len(right_list-1))
    plt.subplots_adjust(wspace=0.35)
    plt.show()

# cv2.imshow('contour_None2', final)
# cv2.imshow('red_obj', red_obj)
# cv2.imshow('yellow_obj', yellow_obj)
cv2.waitKey()


