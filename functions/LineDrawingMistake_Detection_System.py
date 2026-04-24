import psd_tools as pt
from PIL import Image
from psd_tools import PSDImage
import numpy as np
import cv2
from collections import Counter

def LineDrawingMistake_Detection_System(psd, line, padding, mask_size, GAUSSIAN_BLUR):
#仮引数：psdファイル、線画画像、検知箇所の統合判定距離

#PADDING, サイズ最高値から開始し、徐々に減少させる
#一定値毎に検知座標を測り、検知箇所の座標をそれぞれのリストに格納する
#リストに格納された回数の多い座標を優先的に検知箇所とする

#------線画の設定------
    LINE_COLOR = [0, 255, 0, 255]  # 緑
    THRESHOLD_BINARY = 200 # 二値化の閾値
#------輪郭の設定------
    AREA_MIN = 0.0000001 #輪郭面積の最小値  
    AREA_MAX = 10000 #輪郭面積の最大値
    ASPECT_RATIO_THRESHOLD = 2.5 # アスペクト比のしきい値（これより小さいものをゴミ候補とする）
    CIRCULARITY_THRESHOLD = 0.6 # 円形度のしきい値（これより大きいものをゴミ候補とする）

#------円の設定------
    CIRCLE_LARGE = 25
    CIRCLE_MEDIUM = 15
    CIRCLE_SMALL = 5
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

    if GAUSSIAN_BLUR == 'ON':
        line_gray = cv2.GaussianBlur(line_gray, (3, 3), 0) # ガウシアンブラー

    ret, binary = cv2.threshold(line_gray, THRESHOLD_BINARY, 255, cv2.THRESH_BINARY)

    # モルフォロジー変換（オープニング）を追加してノイズ除去
    kernel = np.ones((3,3),np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    #線画の輪郭の検出
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    ) 

    #輪郭の描画
    img_blank = np.ones_like(line_bgr, dtype=np.uint8) * 255
    contours_img = cv2.drawContours(img_blank , contours, -1, color=(0, 0, 255), thickness=2)

    detected_centers = []
    # PADDINGとMASK_SIZEを変化させながら複数回検出を行う
    for p_val in range(padding, 0, -2):
        current_padding = p_val
        current_mask_size = mask_size * current_padding

        #検知箇所の座標計算  
        for i, cnt in enumerate(contours):   
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            area = cv2.contourArea(cnt)   
            if (AREA_MIN < area < AREA_MAX ):
                x, y, w, h = cv2.boundingRect(cnt)

                # アスペクト比の計算
                aspect_ratio = float(w)/h if h > 0 else 0
                # 円形度の計算
                perimeter = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter > 0 else 0

                # 形状によるフィルタリング（ゴミの可能性が高いものを抽出）
                if not (aspect_ratio < ASPECT_RATIO_THRESHOLD and circularity > CIRCULARITY_THRESHOLD):
                    continue

                x1 = max(x - current_padding, 0)
                y1 = max(y - current_padding, 0)
                x2 = min(x + w + current_padding, binary.shape[1] - 1)
                y2 = min(y + h + current_padding, binary.shape[0] - 1)

                roi = binary[y1:y2, x1:x2]

                mask = np.zeros(binary.shape, dtype=np.uint8)
                cv2.drawContours(mask, [cnt], -1, color=255, thickness=-1)
                mask_roi = mask[y1:y2, x1:x2]

                #サイズの閾値()
                if mask_roi.size <= current_mask_size:
                    surrounding = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask_roi))
                    surrounding_count_white = np.sum(surrounding == 255)
                    surrounding_count_black = np.sum(surrounding == 0)

                    #誤検知の判定
                    check = 0
                    if surrounding_count_black + surrounding_count_white > 0:
                        total_pixels = surrounding_count_black + surrounding_count_white
                        black_ratio = surrounding_count_black / total_pixels
                        white_ratio = surrounding_count_white / total_pixels
                        
                        # 基本的な比率による判定（重み付き）
                        if black_ratio <= white_ratio:
                            if white_ratio - black_ratio > 0.6:  # 大きな差がある場合
                                check += 4
                            elif white_ratio - black_ratio > 0.3:  # 中程度の差
                                check += 3
                            else:  # 小さな差
                                check += 2
                        
                        # 白ピクセル比率による詳細判定
                        if white_ratio >= 0.9:
                            check += 3  # 非常に高い白比率
                        elif white_ratio >= 0.8:
                            check += 2.5
                        elif white_ratio >= 0.7:
                            check += 2
                        elif white_ratio >= 0.6:
                            check += 1
                        
                        # 黒ピクセル比率による詳細判定
                        if black_ratio <= 0.05:
                            check += 2  # 非常に低い黒比率
                        elif black_ratio <= 0.1:
                            check += 1.5
                        elif black_ratio <= 0.2:
                            check += 1
                                    
                        if check >= 3: # 信頼性が高いものだけをカウント
                            detected_centers.append(center)

    # 検出された中心点の頻度を数える
    center_counts = Counter(detected_centers)
    
    # 頻度に基づいて円を描画
    # 複数回のチェックで安定して検出されたものをゴミと判断
    # `padding / 4` は、ループ回数のおおよそ半分以上で検出されたものを対象とするための閾値
    detection_threshold = max(1, padding // 4) 
    for center, count in center_counts.items():
        if count >= detection_threshold:
            # 頻度に応じて円の大きさを変えることも可能
            if count >= detection_threshold * 1.5:
                cv2.circle(contours_img, center, CIRCLE_LARGE, CIRCLE_COLOR_RED, CIRCLE_THICKNESS)
            else:
                cv2.circle(contours_img, center, CIRCLE_MEDIUM, CIRCLE_COLOR_ORANGE, CIRCLE_THICKNESS)

    result_img = Image.fromarray(contours_img.astype(np.uint8))
    return result_img
