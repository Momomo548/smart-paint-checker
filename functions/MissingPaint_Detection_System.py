from PIL import Image, ImageDraw
import numpy as np
from scipy.ndimage import label

def MissingPaint_Detection_System(psd, color, tolerance, circle_radius, max_region_size):
#------領域検出の設定------
    COLOR = color  # 検知対象とする色
    TOLERANCE = tolerance  # 色の許容誤差
    MAX_REGION_SIZE = max_region_size  # 塗り漏れと判定する領域のサイズ
    
#------円の設定------
    CIRCLE_RADIUS = circle_radius  # 円の半径
    CIRCLE_THICKNESS = 5  # 円の線の太さ
    
#------色の設定------
    CIRCLE_COLOR_GREEN = (0, 255, 0)  # 検出箇所をマークする円の色

#------色の変換処理------
    if isinstance(COLOR, str) and COLOR.startswith("#"):
        COLOR = COLOR.lstrip('#')
        r = int(COLOR[0:2], 16)
        g = int(COLOR[2:4], 16)
        b = int(COLOR[4:6], 16)
        COLOR = (r, g, b)

#------背景画像の作成------
    psd_width, psd_height = psd.width, psd.height
    background_img = Image.new("RGB", (psd_width, psd_height), COLOR)
    background_img_rgba = background_img.convert("RGBA")
    composite_layer = Image.new("RGBA", (psd_width, psd_height), (0, 0, 0, 0))

#------既存レイヤーの合成------
    for layer in psd.descendants():
        if layer.bbox == (0, 0, 0, 0) or layer.is_group(): # 空のレイヤーはスキップ
            print(f'除外：{layer.name}')
            continue
        is_white = False
        layer_copy = layer.topil()
        layer_copy = np.array(layer_copy)
        is_white = np.all(layer_copy[:, :, :3] == 255)

        if layer.kind == 'pixel' and layer.visible == True:
            if is_white: # 全てのピクセルが白(背景)ならスキップ
                continue
            print(f'layer:{layer.name} ->{is_white}')
            left, top, right, bottom = layer.bbox
            composite_layer.paste(layer.composite(), (left, top), layer.composite())

            
    composite_result = composite_layer.convert("RGBA")
    background_img_rgba.paste(composite_result, (0, 0), composite_result)
    result_image = background_img_rgba


#-----蛍光色領域の検出------
    # アルファチャンネルを考慮した色検出
    img_array = np.array(result_image)
    alpha_channel = img_array[:, :, 3]
    rgb_array = img_array[:, :, :3]
    
    r, g, b = COLOR
    
    # TOLERANCEを自動調整するループ
    min_required_features = 200 # 最低限必要な領域数
    max_tolerance = 200 # TOLERANCE上限値
    tolerance_step = 1         # TOLERANCEの増加幅
    
    current_tolerance = TOLERANCE
    num_features = 0
    detection_mask = None
    
    # 色検出のみ（アルファ除外）
    color_detection = np.all(np.abs(rgb_array - np.array([r, g, b])) < current_tolerance, axis=-1)
    
    # アルファ値が0でない（透明でない）ピクセルのみを対象
    detection_mask = color_detection & (alpha_channel > 5)
    
    # 連結領域のラベリング
    labeled_array, num_features = label(detection_mask)




#------検知箇所の座標計算------
    draw = ImageDraw.Draw(result_image)
    for region_num in range(2, num_features + 1):
        region_mask = (labeled_array == region_num)
        region_size = np.sum(region_mask)


        if region_size <= MAX_REGION_SIZE:
            region_indices = np.argwhere(region_mask)
            y_min, x_min = region_indices.min(axis=0)
            y_max, x_max = region_indices.max(axis=0)
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            draw.ellipse(
                (center_x - CIRCLE_RADIUS, center_y - CIRCLE_RADIUS,
                 center_x + CIRCLE_RADIUS, center_y + CIRCLE_RADIUS),
                outline=CIRCLE_COLOR_GREEN, width=CIRCLE_THICKNESS)
    
    return result_image
