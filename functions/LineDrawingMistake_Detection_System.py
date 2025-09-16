from PIL import Image
import numpy as np
import cv2

def LineDrawingMistake_Detection_System(psd, line, threshold):
#仮引数：psdファイル、線画画像、検知箇所の統合判定距離
    LINE_COLOR = [0, 255, 0, 255]  # 緑
    THRESHOLD_BINARY = 200
    AREA_MIN = 1
    AREA_MAX = 10000
    CIRCLE_RADIUS = 50
    CIRCLE_THICKNESS = 10
    CIRCLE_COLOR = (255, 0, 0)

    #線画の色を緑色に変更
    line_np = np.array(line)
    line_color = LINE_COLOR
    #線画部分（アルファ値が0でない部分）を緑色に変更
    mask = line_np[:, :, 3] > 0
    line_np[mask] = line_color
    #RGBAからBGRに変換
    line_bgr = cv2.cvtColor(line_np, cv2.COLOR_RGBA2BGR)
    #グレースケールに変換
    line_gray = cv2.cvtColor(line_bgr, cv2.COLOR_BGR2GRAY)
    #二値化（150を閾値に設定）
    ret, binary = cv2.threshold(line_gray, THRESHOLD_BINARY, 255, cv2.THRESH_BINARY)
    #線画の輪郭の検出
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )
    #輪郭の描画
    img_blank = np.ones_like(line_bgr, dtype=np.uint8) * 255
    contours_img = cv2.drawContours(img_blank , contours, -1, color=(0, 0, 255), thickness=4)
    #検知箇所の座標
    circle_centers = []
    #近い線画を合成後の検知箇所の座標
    filtered_centers = []
    #検出数
    count = 0
    #検知箇所の座標計算
    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        area = cv2.contourArea(cnt)
        if (AREA_MIN < area < AREA_MAX ):
            circle_centers.append(center)
    #重複検出と円描画
    for i, center1 in enumerate(circle_centers):
        check = False
        for j, center2 in enumerate(filtered_centers):
            distance = np.linalg.norm(np.array(center1) - np.array(center2))
            if distance < threshold:
                check = True
                break    
        if not check:
            filtered_centers.append(center1)
    for center in filtered_centers:
        cv2.circle(contours_img, center, CIRCLE_RADIUS, CIRCLE_COLOR, CIRCLE_THICKNESS)
        count+=1
    result_img = Image.fromarray(contours_img.astype(np.uint8))
    return result_img