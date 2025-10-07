from PIL import Image
import numpy as np
import cv2

def LineDrawingMistake_Detection_System(psd, line, padding, mask_size, mistake_threshold_black, mistake_threshold_white):
#仮引数：psdファイル、線画画像、検知箇所の統合判定距離

#------線画の設定------
    LINE_COLOR = [0, 255, 0, 255]  # 緑
    THRESHOLD_BINARY = 200 # 二値化の閾値
#------輪郭の設定------
    AREA_MIN = 0.0000001 #輪郭面積の最小値  
    AREA_MAX = 10000 #輪郭面積の最大値
    PADDING = padding #輪郭の周囲に付与する余白
    MASK_SIZE = mask_size * padding #マスクのサイズ
    MISTAKE_THRESHOLD_BLACK = mistake_threshold_black  #どの大きさまで黒ピクセルを許容するか
    MISTAKE_THRESHOLD_WHITE = mistake_threshold_white #どの大きさまで白ピクセルを許容するか

#------円の設定------
    CIRCLE_LARGE = 50
    CIRCLE_MEDIUM = 30
    CIRCLE_SMALL = 25
    CIRCLE_THICKNESS = 5
#------色の設定------
    CIRCLE_COLOR_RED = (255, 0, 0)
    CIRCLE_COLOR_GREEN = (0, 255, 0)
    CIRCLE_COLOR_BLUE = (0, 0, 255)
    CIRCLE_COLOR_ORANGE = (255, 165, 0)
    CIRCLE_COLOR_YELLOW = (255, 255, 0)

    #線画の色を緑色に変更
    line_np = np.array(line)
    #線画部分（アルファ値が0でない部分）を緑色に変更
    mask = line_np[:, :, 3] > 0
    line_np[mask] = LINE_COLOR

    #RGBA -> BGR -> グレースケール変換
    line_bgr = cv2.cvtColor(line_np, cv2.COLOR_RGBA2BGR)
    line_gray = cv2.cvtColor(line_bgr, cv2.COLOR_BGR2GRAY)

    ret, binary = cv2.threshold(line_gray, THRESHOLD_BINARY, 255, cv2.THRESH_BINARY)
    #線画の輪郭の検出
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )

    #輪郭の描画
    img_blank = np.ones_like(line_bgr, dtype=np.uint8) * 255
    contours_img = cv2.drawContours(img_blank , contours, -1, color=(0, 0, 255), thickness=2)

    #検知箇所の座標計算  
    for i, cnt in enumerate(contours):   
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        area = cv2.contourArea(cnt)   
        if (AREA_MIN < area < AREA_MAX ):
            x, y, w, h = cv2.boundingRect(cnt)

            x1 = max(x - PADDING, 0)
            y1 = max(y - PADDING, 0)
            x2 = min(x + w + PADDING, binary.shape[1] - 1)
            y2 = min(y + h + PADDING, binary.shape[0] - 1)
            roi = binary[y1:y2, x1:x2]

            mask = np.zeros(binary.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, color=255, thickness=-1)
            mask_roi = mask[y1:y2, x1:x2]

            #サイズの閾値()
            if mask_roi.size <= MASK_SIZE:
                surrounding = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask_roi))
                surrounding_count_white = np.sum(surrounding == 255)
                surrounding_count_black = np.sum(surrounding == 0)

                #誤検知の判定
                check = 0
                total_pixels = surrounding_count_black + surrounding_count_white
                black_ratio = surrounding_count_black / total_pixels
                white_ratio = surrounding_count_white / total_pixels
                
                if black_ratio <= 0.1:  # 黒ピクセルが10%以下
                    check += 4
                elif white_ratio >= 0.7:  # 白ピクセルが80%以上
                    check += 3
                if surrounding_count_black <= MISTAKE_THRESHOLD_BLACK:
                    check += 2
                elif surrounding_count_white >= MISTAKE_THRESHOLD_WHITE:
                    check += 1
                
                (cx, cy), _ = cv2.minEnclosingCircle(cnt)
                center = (int(cx), int(cy))
                # 番号表示用の座標（円の右上）
                text_pos = (center[0] + 15, center[1] - 10)
                if check >= 5:
                    cv2.circle(contours_img, center, CIRCLE_LARGE, CIRCLE_COLOR_RED, CIRCLE_THICKNESS)
                    cv2.putText(contours_img, str(i+1), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, CIRCLE_COLOR_RED, 2)
                elif 2 <= check < 5:
                    cv2.circle(contours_img, center, CIRCLE_MEDIUM, CIRCLE_COLOR_ORANGE, CIRCLE_THICKNESS)
                    cv2.putText(contours_img, str(i+1), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, CIRCLE_COLOR_ORANGE, 2)
                else:
                    cv2.circle(contours_img, center, CIRCLE_SMALL, CIRCLE_COLOR_GREEN, CIRCLE_THICKNESS)
                    cv2.putText(contours_img, str(i+1), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, CIRCLE_COLOR_YELLOW, 2)

    result_img = Image.fromarray(contours_img.astype(np.uint8))
    return result_img
