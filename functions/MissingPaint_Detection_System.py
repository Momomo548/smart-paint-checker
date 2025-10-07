from PIL import Image, ImageDraw
import numpy as np
from scipy.ndimage import label

def MissingPaint_Detection_System(psd, color, tolerance, circle_radius, max_region_size):
#仮引数：psdファイル、検知箇所カラー、色の許容誤差、円の半径、塗り漏れと判定する領域のサイズ

#------領域検出の設定------
    COLOR = color  # 検知対象とする色
    TOLERANCE = tolerance  # 色の許容誤差
    MAX_REGION_SIZE = 10000  # 塗り漏れと判定する領域のサイズ
    
#------円の設定------
    CIRCLE_RADIUS = circle_radius  # 円の半径
    CIRCLE_THICKNESS = 10  # 円の線の太さ
    
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

#------既存レイヤーの合成------
    existing_layers = psd.topil()
    result_image = Image.alpha_composite(background_img_rgba, existing_layers.convert("RGBA"))

#------蛍光色領域の検出------
    img_array = np.array(result_image.convert("RGB"))
    r, g, b = COLOR
    mask = np.all(np.abs(img_array - np.array([r, g, b])) <= TOLERANCE, axis=-1)

#------連結領域のラベリング------
    labeled_array, num_features = label(mask)

#------検知箇所の座標計算------
    draw = ImageDraw.Draw(result_image)
    for region_num in range(1, num_features + 1):
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