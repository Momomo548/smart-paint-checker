from PIL import Image
import numpy as np
import cv2

def Layers_Contour_System(psd, layers, threshold, color, contour_size, layer_display_switching, background_display_switching, background):
#仮引数：表示するレイヤー画像、二値化の閾値、輪郭線のカラー、輪郭線の幅、レイヤー画像の表示切替、背景の表示切替、背景カラー
    #輪郭線のカラーと背景カラーをRGB値へ変換
    if isinstance(color, str) and color.startswith("#"):
        color = color.lstrip('#')
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
        color = (b, g, r)
    if isinstance(background, str) and background.startswith("#"):
        background = background.lstrip('#')
        r = int(background[0:2], 16)
        g = int(background[2:4], 16)
        b = int(background[4:6], 16)
        background = (r, g, b)
    #出力画像の生成
    background = Image.new("RGBA", psd.composite().size, (background[0], background[1], background[2], 255))
    #レイヤー画像をNumpy配列に変換
    layers_np = np.array(layers, dtype=np.uint8)
    #RGBAからBGRに変換
    layers_bgr = cv2.cvtColor(layers_np, cv2.COLOR_RGBA2BGR)
    #グレースケールに変換
    layers_gray = cv2.cvtColor(layers_bgr, cv2.COLOR_BGR2GRAY)
    #二値化
    ret, binary = cv2.threshold(layers_gray, threshold, 255, cv2.THRESH_BINARY)
    #輪郭を検出
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #全て白または黒のカラー画像を作成
    if color == (255, 255, 255):
        #背景は黒（輪郭は白）
        img_blank = np.zeros_like(layers_bgr, dtype=np.uint8)
    else:
        #背景は白（輪郭は白以外）
        img_blank = np.ones_like(layers_bgr, dtype=np.uint8) * 255
    #輪郭を描画
    contour_img = cv2.drawContours(img_blank, contours, -1, color, contour_size)
    #OpenCV形式からPillow形式に変換
    result_img = Image.fromarray(cv2.cvtColor(contour_img, cv2.COLOR_BGRA2RGBA))
    #輪郭以外を透過させる（R, G, B がすべて 255）の場合、アルファ値を 0 に設定
    datas = result_img.getdata()
    new_data = []
    if color == (255, 255, 255):
        for item in datas:
            #輪郭以外（黒）を透過させる
            if item[0] == 0 and item[1] == 0 and item[2] == 0:
                new_data.append((255, 255, 255, 0))
            else:
                new_data.append(item)
        result_img.putdata(new_data)
    else:
        for item in datas:
            #輪郭以外（白）を透過させる
            if item[0] == 255 and item[1] == 255 and item[2] == 255:
                new_data.append((255, 255, 255, 0))
            else:
                new_data.append(item)
        result_img.putdata(new_data)
    #輪郭とレイヤー画像を合成
    if layer_display_switching == "表示":
        result_img.paste(layers, (psd.left, psd.top), layers)
    if background_display_switching == "表示":
        background.paste(result_img, (psd.left, psd.top), result_img)
        result_img = background
    return result_img