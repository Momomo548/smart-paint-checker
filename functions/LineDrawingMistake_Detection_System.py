import psd_tools as pt
from PIL import Image
from psd_tools import PSDImage
import numpy as np
import cv2
from collections import Counter

def LineDrawingMistake_Detection_System(psd, line, padding, mask_size, contour_color, circle_color):
    #仮引数：psdファイル、線画画像、検知箇所の統合判定距離

    #PADDING, サイズ最高値から開始し、徐々に減少させる
    #一定値毎に検知座標を測り、検知箇所の座標をそれぞれのリストに格納する
    #リストに格納された回数の多い座標を優先的に検知箇所とする

    #------線画の設定------
        THRESHOLD_BINARY = 200 # 二値化の閾値
    #------輪郭の設定------
        AREA_MIN = 0.0000001 #輪郭面積の最小値  
        AREA_MAX = 10000 #輪郭面積の最大値
        ASPECT_RATIO_THRESHOLD = 2.0 # アスペクト比のしきい値（これより小さいものをゴミ候補とする）
        CIRCULARITY_THRESHOLD = 0.65 # 円形度のしきい値（これより大きいものをゴミ候補とする）
        GAUSSIAN_BLUR = False
    #------円の設定------
        CIRCLE_LARGE = 25
        CIRCLE_MEDIUM = 15
        CIRCLE_SMALL = 5
        CIRCLE_THICKNESS = 5
    #------色の設定------
        RED = (255, 0, 0)
        GREEN = (0, 255, 0)
        BLUE = (0, 0, 255)
        COLORS = {'red': RED, 
                  'green': GREEN, 
                  'blue': BLUE,}
        
        CIRCLE_COLOR = COLORS.get(circle_color.lower(), RED)
        LINE_COLOR_RGB = COLORS.get(contour_color.lower(), GREEN)
        LINE_COLOR = (*LINE_COLOR_RGB, 255) # RGBAに変換

        #線画の色を緑色に変更
        line_np = np.array(line)
        #線画部分（アルファ値が0でない部分）を緑色に変更
        if line_np.shape[2] == 4:
            mask = line_np[:, :, 3] > 0
            if np.any(mask):
                line_np[mask] = LINE_COLOR

        #RGBA -> BGR -> グレースケール変換
        line_bgr = cv2.cvtColor(line_np, cv2.COLOR_RGBA2BGR)
        line_gray = cv2.cvtColor(line_bgr, cv2.COLOR_BGR2GRAY)


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
        contours_img = cv2.drawContours(img_blank.copy() , contours, -1, color=LINE_COLOR, thickness=2)

        # --- 変更点：最初に特徴量検出で候補を絞り込む ---
        candidate_contours = []
        akaze = cv2.AKAZE_create()

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if not (AREA_MIN < area < AREA_MAX):
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            
            # 形状によるフィルタリング
            aspect_ratio = float(w)/h if h > 0 else 0
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter > 0 else 0
            if not (aspect_ratio < ASPECT_RATIO_THRESHOLD and circularity > CIRCULARITY_THRESHOLD):
                continue

            # 特徴量検出
            mask = np.zeros(binary.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, color=255, thickness=-1)
            gray_roi = line_gray[y:y+h, x:x+w]
            mask_roi = mask[y:y+h, x:x+w]
            
            keypoints, descriptors = akaze.detectAndCompute(gray_roi, mask_roi)
            keypoint_count = len(keypoints)

            avg_response = 0
            if keypoint_count > 0:
                avg_response = np.mean([kp.response for kp in keypoints])

            # 輪郭の面積と特徴量に基づいた、より厳格な条件分岐で候補を絞る
            is_potential_noise = False
            if area < 5:
                if keypoint_count == 0:
                    is_potential_noise = True
            elif area < 20:
                if keypoint_count <= 1 and avg_response < 0.001:
                    is_potential_noise = True
            else:
                if keypoint_count <= 0:
                    is_potential_noise = True
            
            if is_potential_noise:
                candidate_contours.append(cnt)
                #print(f'候補 {i}: 特徴点 {keypoint_count}, avg_response {avg_response:.4f}, area {area}')

        # 候補輪郭に対してのみ、周辺ピクセルのチェックを行う
        detected_centers = []
        for p_val in range(padding, 0, -1):
            current_padding = p_val
            current_mask_size = mask_size * current_padding

            #検知箇所の座標計算  
            for i, cnt in enumerate(candidate_contours):   
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                center = (int(x), int(y))
                area = cv2.contourArea(cnt)
                
                x, y, w, h = cv2.boundingRect(cnt)

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
                    surrounding_count_white = np.sum(surrounding >= 240)
                    surrounding_count_black = np.sum(surrounding == 0)

                    # 周辺ピクセルによる初期スコア計算
                    check = 0
                    total_pixels = surrounding_count_black + surrounding_count_white
                    if total_pixels > 0:
                        white_ratio = surrounding_count_white / total_pixels
                        black_ratio = surrounding_count_black / total_pixels

                        # 判定基準1: 周囲が白いほどゴミの可能性が高い (重み: 7)
                        score1 = white_ratio * 7.0

                        # 判定基準2: 周囲に黒い部分が少ないほどゴミの可能性が高い (重み: 3)
                        score2 = (1.0 - black_ratio) * 3.0
                        
                        # 合計スコアを0-10の範囲で算出
                        check = score1 + score2
                    
                    # スコアが6以上の物を検出点とする
                    if check >= 6:
                        detected_centers.append(center)
                        #print(f'検出 {i} at {center}: p_val={p_val}, score={check:.2f}')


        # 検出された中心点の頻度を数える
        center_counts = Counter(detected_centers)
        #print(f'\n検出された中心点の頻度: {center_counts}')
        
        # 頻度に基づいて円を描画
        # 複数回のチェックで安定して検出されたものをゴミと判断
        # `padding / 4` は、ループ回数のおおよそ半分以上で検出されたものを対象とするための閾値
        #detection_threshold = max(1, padding // 4)
        
        # 円が描画された中心座標を保存するリスト
        detected_circles = []
        
        for center, count in center_counts.items():

            if count >= padding:
                cv2.circle(contours_img, center, CIRCLE_LARGE, CIRCLE_COLOR, CIRCLE_THICKNESS)
                detected_circles.append({'x': center[0], 'y': center[1]})


        result_img = Image.fromarray(contours_img.astype(np.uint8))

        return result_img