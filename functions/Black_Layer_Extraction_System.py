from PIL import Image
import numpy as np
import cv2
from psd_tools import PSDImage

#psds フォルダにて開発中
#終了次第コピペして実装

def Black_Layer_Extraction_System(psd, black_threshold):
    """
    PSDファイルのすべてのレイヤーを探索し、黒色のみで構成されているレイヤーを抽出する関数
    
    Parameters:
    -----------
    psd : PSDImage
        解析対象のPSDファイルオブジェクト
    black_threshold : int (デフォルト: 50)
        黒色と判定する閾値（RGB各値がこの値以下なら黒色と判定）
    
    Returns:
    --------
    black_layers : dict
        黒色のみのレイヤー情報を含む辞書
        {
            'layer_list': [(レイヤー名, レイヤーオブジェクト, 黒色ピクセル率), ...],
            'count': 黒色のみのレイヤー数,
            'total_count': 全レイヤー数
        }
    """
    
    black_layers = {
        'layer_list': [],
        'count': 0,
        'total_count': 0
    }
    
    # PSDのすべてのレイヤーを探索
    all_layers = list(psd.descendants())
    black_layers['total_count'] = len([layer for layer in all_layers if not layer.is_group()])
    
    for layer in all_layers:
        # グループと空のレイヤーはスキップ
        if layer.is_group() or layer.bbox == (0, 0, 0, 0):
            continue
        
        # レイヤーを PIL Image に変換
        layer_img = layer.topil()
        
        if layer_img is None:
            continue
        
        # numpy配列に変換
        layer_np = np.array(layer_img)
        
        # アルファチャンネルがある場合のみ処理（透明度チェック）
        if layer_np.shape[2] == 4:
            alpha = layer_np[:, :, 3]
            # 透明度が0より大きいピクセルのみを対象
            valid_pixels = alpha > 0
        else:
            # アルファチャンネルがない場合は全ピクセルを対象
            valid_pixels = np.ones(layer_np.shape[:2], dtype=bool)
        
        # 有効なピクセルが存在しない場合はスキップ
        if valid_pixels.sum() == 0:
            continue
        
        # RGB値をチェック（黒色判定：R,G,Bすべてが閾値以下）
        r = layer_np[:, :, 0]
        g = layer_np[:, :, 1]
        b = layer_np[:, :, 2]
        
        # 黒色ピクセルのマスク（有効なピクセル内で黒色のもの）
        black_pixels = (r <= black_threshold) & (g <= black_threshold) & (b <= black_threshold) & valid_pixels
        
        # 有効なピクセル内での黒色ピクセル率を計算
        black_pixel_rate = black_pixels.sum() / valid_pixels.sum()
        
        # ほぼすべてのピクセルが黒色（98%以上）なら黒色レイヤーと判定
        if black_pixel_rate >= 0.98:
            black_layers['layer_list'].append({
                'name': layer.name,
                'layer': layer,
                'black_pixel_rate': black_pixel_rate,
                'bbox': layer.bbox,
                'visible': layer.visible,
                'kind': layer.kind
            })
            black_layers['count'] += 1
    
    return black_layers


def Get_Black_Layer_Images(psd, psd_composite_size, black_threshold=50):
    """
    黒色のみで構成されているレイヤーの複合画像を生成する関数
    
    Parameters:
    -----------
    psd : PSDImage
        解析対象のPSDファイルオブジェクト
    psd_composite_size : tuple
        PSD画像のサイズ (width, height)
    black_threshold : int (デフォルト: 50)
        黒色と判定する閾値
    
    Returns:
    --------
    composite_image : PIL Image (RGBA)
        黒色レイヤーを合成した画像
    black_layers_info : dict
        黒色レイヤーの情報
    """
    
    # 黒色レイヤーの抽出
    black_layers_info = Black_Layer_Extraction_System(psd, black_threshold)
    
    # 複合画像の作成（透過背景）
    composite_image = Image.new("RGBA", psd_composite_size, (255, 255, 255, 0))
    
    # 黒色レイヤーを合成
    for layer_info in black_layers_info['layer_list']:
        layer = layer_info['layer']
        layer_img = layer.composite()
        
        if layer_img is not None:
            left, top, right, bottom = layer.bbox
            composite_image.paste(layer_img, (left, top), layer_img)
    
    return composite_image, black_layers_info

