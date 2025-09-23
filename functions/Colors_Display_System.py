from PIL import Image
import numpy as np

def Colors_Display_System(psd, colors, thresholds, background_display_switching, background):
#仮引数：psdファイル、表示するカラーリスト、各カラーの許容範囲リスト、背景の表示切替、背景カラー
    #出力画像の生成
    if background_display_switching == "非表示":
        result_img = Image.new("RGBA", psd.composite().size, (255, 255, 255, 0))
    else: 
        #背景カラーをRGB値へ変換
        if isinstance(background, str) and background.startswith("#"):
            background = background.lstrip('#')
            r = int(background[0:2], 16)
            g = int(background[2:4], 16)
            b = int(background[4:6], 16)
            background = (r, g, b)
        result_img = Image.new("RGBA", psd.composite().size, (background[0], background[1], background[2], 255))
    #表示するカラー数だけ繰り返し
    for (color, threshold) in zip(colors, thresholds):
        #表示するカラーをRGB値へ変換
        if isinstance(color, str) and color.startswith("#"):
            color = color.lstrip('#')
            r = int(color[0:2], 16)
            g = int(color[2:4], 16)
            b = int(color[4:6], 16)
        #レイヤーを解析
        for layer in psd.descendants():
            if not layer.is_group():
                layer_img = layer.topil()    
                if layer_img is None:
                    continue
                layer_np = np.array(layer_img)
                #レイヤ画像が透過度を持つかを判定
                if layer_np.shape[2] == 4:
                    alpha = layer_np[:, :, 3]
                    #透明度が０より大きいピクセルを取得
                    valid_pixels = alpha > 0
                else:
                    #透過度を持たない場合は全てのピクセルを取得
                    valid_pixels = np.ones(layer_np.shape[:2], dtype=bool)
                #指定カラーとの差を計算し、閾値内の色を一致とみなす
                mask = (np.abs(layer_np[:, :, 0] - r) <= threshold[0]) & \
                       (np.abs(layer_np[:, :, 1] - g) <= threshold[1]) & \
                       (np.abs(layer_np[:, :, 2] - b) <= threshold[2]) & valid_pixels
                alpha_channel = np.where(mask, 255, 0).astype(np.uint8)
                result_layer = np.dstack((layer_np[:, :, :3], alpha_channel))
                layer_img = Image.fromarray(result_layer, "RGBA") 
                result_img.paste(layer_img, (layer.left, layer.top), layer_img)
    return result_img