from PIL import Image, ImageDraw
import numpy as np
from scipy.ndimage import label

def MissingPaint_Detection_System(psd, color, tolerance, circle_radius, max_region_size):
#仮引数：psdファイル、検知箇所カラー、色の許容誤差、円の半径、塗り漏れと判定する領域のサイズ
    if isinstance(color, str) and color.startswith("#"):
        color = color.lstrip('#')
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
        color = (r, g, b)
    #背景画像を作成
    psd_width, psd_height = psd.width, psd.height
    background_img = Image.new("RGB", (psd_width, psd_height), color)
    background_img_rgba = background_img.convert("RGBA")
    #既存のPSDファイルのレイヤーを合成
    existing_layers = psd.topil()
    #背景と合成して最終画像を作成
    result_image = Image.alpha_composite(background_img_rgba, existing_layers.convert("RGBA"))

    #蛍光色の領域を検出
    img_array = np.array(result_image.convert("RGB"))
    r, g, b = color
    mask = np.all(np.abs(img_array - np.array([r, g, b])) <= tolerance, axis=-1)
    #連結した領域を結合し、ラベリング
    labeled_array, num_features = label(mask)
    #領域サイズの確認と円の描画
    draw = ImageDraw.Draw(result_image)
    for region_num in range(1, num_features + 1):
        region_mask = (labeled_array == region_num)
        region_size = np.sum(region_mask)
        if region_size <= 10000:
            region_indices = np.argwhere(region_mask)
            y_min, x_min = region_indices.min(axis=0)
            y_max, x_max = region_indices.max(axis=0)
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            draw.ellipse(
                (center_x - circle_radius, center_y - circle_radius,
                 center_x + circle_radius, center_y + circle_radius),
                outline=(0, 255, 0), width=10)
    return result_image