#=================================================================
# インポート
#=================================================================
import const
import streamlit as st
from streamlit_option_menu import option_menu
import psd_tools
from PIL import Image, ImageDraw
import numpy as np
import cv2
from scipy.ndimage import label
from io import BytesIO
import zipfile
import hashlib
from streamlit_image_select import image_select

from functions import (
    MissingPaint_Detection_System,
    LineDrawingMistake_Detection_System,
    Layers_Contour_System,
    Colors_Display_System
)


#=================================================================
# ページ、レイアウト設定
# ・SET_PAGE_CONFIG：ページの設定
# ・HIDE_ST_STYLE：レイアウトの調節
# ・OPTION_MENU_CONFIG：タブの設定
# ・st.markdown：markdown設定
#=================================================================
st.set_page_config(**const.SET_PAGE_CONFIG)
st.markdown(const.HIDE_ST_STYLE, unsafe_allow_html=True)
selected = option_menu(**const.OPTION_MENU_CONFIG)
st.markdown('''
<style>.my-text {
    color: white;
    font-size: 24px;
    background-color: #008080;
    padding: 10px;
    border-radius: 10px;
}</style>

<style>.box {
    background:#c4d9ff;
    border-left:#4e7bcc 5px solid;
    padding:2px;
    width: 300px;
    font-size: 20px;
    text-align: center; 
}</style>

<style>.box2{
    background:#c4d9ff;
    border-left:#4e7bcc 5px solid;
    padding:2px;
    width: 400px;
    font-size: 20px;
    text-align: center; 
}</style>''', unsafe_allow_html=True)
    
#=================================================================
# HOME画面
#=================================================================
if selected == 'HOME':
    st.markdown('''
    <p class='my-text'>デジタルイラスト制作における課題と本研究の目的</p>
    
    ゲーム会社など、デジタルイラストを大量に扱う企業では、外部のイラスト制作会社に業務を委託することがある。  
    そして、委託を受けた制作会社は完成したイラストを納品する前に、品質を保証するため「塗り漏れ」「はみ出し」「消し忘れ」などのミスがないかを細部までチェックする必要がある。 
    しかし、これらのチェック作業は従来、手作業で行われており、手間がかかるという課題が指摘されている。  
    そこで本研究では、画像処理技術を用いて「塗り漏れ」「はみ出し」「消し忘れ」ミスを自動的に検知、または発見を支援する方式を提案する。  
    さらに、これらの方式に基づいてイラストの着色チェックツール「Smart Paint Checker」を開発することで、ミスチェック作業の効率化とクリエイターの負担軽減を図ることを目的とする。
    ''', unsafe_allow_html=True)
    st.image('./Images/miss.png', width=700) 
    
    st.markdown('''
    <p class='my-text'>開発体制</p>
    
    ##### 組織名
    東京電機大学 システムデザイン工学部 情報システム工学科 マルチメディアコンピューティング研究室
    
    ##### メンバー
    ・石川 陸斗：全体システム、指定カラー表示システム、指定レイヤー輪郭表示システム 開発担当  
    ・染谷 学玖：塗り漏れ検知システム 開発担当  
    ・中野 爽一朗：消し忘れ検知システム 開発担当
    ''', unsafe_allow_html=True)
    link = '[研究室へのご連絡はこちらから](https://mclab.jp/?page_id=368)'
    st.markdown(link, unsafe_allow_html=True)
    
    st.markdown('''
    <p class='my-text'>使用したイラスト素材</p>
    
    ##### ユニティちゃん
    <div><img src="https://unity-chan.com/images/imageLicenseLogo.png" alt="ユニティちゃんライセンス"><p>
    
    『<a href="https://unity-chan.com/contents/license_jp/" target="_blank">ユニティちゃんライセンス</a>』  
    『<a href="https://unity-chan.com/contents/guideline/" target="_blank">キャラクター利用のガイドライン</a>』
    ''', unsafe_allow_html=True)
    
#=================================================================
# 機能詳細と使用例画面
#=================================================================
if selected == '機能詳細と使用例':
    #機能１
    st.markdown('''
    <p class='my-text'>機能１：指定カラー表示システム</p>
    <p class='box'>機能内容</p>
    
    イラストファイルとカラーを指定することで、イラスト内でそのカラーが使用されている箇所のみを表示する機能である。  
    また、塗りの濃さが異なる場合を考慮し、RGB各値に許容範囲（±）を指定することや複数のカラーを同時に指定することも可能である。  
    ''', unsafe_allow_html=True)
    st.image('./Images/color_display.png', width=700) 
    st.markdown('''<p class='box'>使用例</p>''', unsafe_allow_html=True)
    st.image('./Images/color_display_ex.png', width=700) 
    
    #機能２
    st.markdown('''
    <p class='my-text'>機能２：指定レイヤー輪郭表示システム</p>
    <p class='box'>機能内容</p>
    
    イラストファイルとレイヤーを指定することで、そのレイヤー画像と輪郭を表示する機能である。  
    また、レイヤー画像を非表示にして輪郭のみを表示することや複数のレイヤーを同時に指定することも可能である。
    ''', unsafe_allow_html=True)
    st.image('./Images/layer_display.png', width=700) 
    st.markdown('''<p class='box'>使用例</p>''', unsafe_allow_html=True)
    st.image('./Images/layer_display_ex1.png', width=700) 
    st.image('./Images/layer_display_ex2.png', width=700) 
    
    #機能３
    st.markdown('''
    <p class='my-text'>機能３：塗り漏れ検知システム</p>
    <p class='box'>機能内容</p>
    
    イラストファイルを指定することで、新たな色の背景を追加し、透過した塗り漏れを検知して円で囲む機能である。  
    また、背景色や円の大きさ、塗り漏れと判定する許容値を指定することも可能である。
    ''', unsafe_allow_html=True)
    st.image('./Images/missing_paint.png', width=700) 
    
    #機能４
    st.markdown('''
    <p class='my-text'>機能４：消し忘れ検知システム</p>
    <p class='box'>機能内容</p>
    
    イラストファイルと線画レイヤーを指定することで、線画の消し忘れ箇所を検知し円で囲む機能である。  
    また、検知機能の強さを指定可能であり、値が低いほど厳しく検知出来る
    ''', unsafe_allow_html=True)
    st.image('./Images/missing_lines.png', width=700) 
    
#=================================================================
# ツールを使用する画面
#=================================================================
if selected == 'ツールを使用する':
    #========================================
    # 「機能の選択」
    #========================================
    st.markdown('''
    <p class='my-text'>機能の選択</p>
    
    #### 以下から使用したい機能を選択してください。（:red[複数同時実行可能]）''', unsafe_allow_html=True)
    
    functions = []
    if st.checkbox('指定カラー表示システム'):
        functions.append('Colors_Display_System')
    st.write('<span style="color:red;background:pink">：イラスト内で指定したカラーが使用されている箇所を表示する</span>'
    ,unsafe_allow_html=True)
    if st.checkbox('指定レイヤー輪郭表示システム'):
        functions.append('Layers_Contour_System')
    st.write('<span style="color:red;background:pink">：指定したレイヤー画像の輪郭を抽出する</span>'
    ,unsafe_allow_html=True)
    if st.checkbox('塗り漏れ検知システム'):
        functions.append('MissingPaint_Detection_System')
    st.write('<span style="color:red;background:pink">：イラスト内の「塗り漏れ」ミスを検知する</span>'
    ,unsafe_allow_html=True)
    if st.checkbox('消し忘れ検知システム'):
        functions.append('LineDrawingMistake_Detection_System')
    st.write('<span style="color:red;background:pink">：線画の「消し忘れ」ミスを検知する</span>'
    ,unsafe_allow_html=True)
    
    #========================================
    # 「ファイルの選択」
    #========================================
    st.markdown('''<p class='my-text'>ファイルの選択</p>''', unsafe_allow_html=True)
    #セッション変数の初期化（エラー判定・レイヤー一覧用）
    if "last_file_hash1" not in st.session_state:
        st.session_state.last_file_hash1 = None
    if "last_file_hash2" not in st.session_state:
        st.session_state.last_file_hash2 = None
    if "error_flag" not in st.session_state:
        st.session_state.error_flag = None
    if "name_list" not in st.session_state:
        st.session_state.variable_value = []
    if "img_list" not in st.session_state:
        st.session_state.variable_value = []
    if "layer_number" not in st.session_state:
        st.session_state.variable_value = 0
    if "psd_cache" not in st.session_state:
        st.session_state.psd_cache = None
    if "psd_composite_cache" not in st.session_state:
        st.session_state.psd_composite_cache = None
    #ファイルのアップロード
    uploaded_file = st.file_uploader('#### 使用するPSDファイルを選択してください。', type='psd')
    if uploaded_file is not None:
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text('ファイルの読み込み中...')
        progress_bar.progress(20)
        #ファイルのハッシュ値を計算
        uploaded_file.seek(0)
        file_hash = hashlib.sha256(uploaded_file.read()).hexdigest()
        uploaded_file.seek(0)
        current_file_hash = file_hash
        progress_bar.progress(50)
        
        # キャッシュされたpsdを使用するか、新規に読み込む
        if (st.session_state.last_file_hash1 != current_file_hash or 
            st.session_state.psd_cache is None):
            status_text.text('PSDファイルの処理中...')
            progress_bar.progress(60)

            psd = psd_tools.PSDImage.open(uploaded_file)
            psd_composite = psd.composite()
            st.session_state.psd_cache = psd
            st.session_state.psd_composite_cache = psd_composite
        else:
            psd = st.session_state.psd_cache
            psd_composite = st.session_state.psd_composite_cache

        progress_bar.progress(70)

        #========================================
        # 「エラー判定」
        #========================================
        #入力ファイルが変更されたかを判定
        if st.session_state.last_file_hash1 != current_file_hash:
            #エラーイラストの判定
            psd_pixels = psd_composite.getdata()
            total_pixels = len(psd_pixels)
            empty_pixels = sum(1 for psd_pixel in psd_pixels if len(psd_pixel) > 3 and psd_pixel[3] == 0)
            #セッション変数を更新（エラー判定用）
            if empty_pixels == total_pixels:
                st.session_state.error_flag = 'エラー'
            else:
                st.session_state.error_flag = '正常'
            st.session_state.last_file_hash1 = current_file_hash
        status_text.text('画像を検証中...')
        progress_bar.progress(80)
        #判定結果の出力
        if st.session_state.error_flag == 'エラー':
            st.error('透明な画像が入力されました。画像を変更してください。')
        else:
            progress_bar.progress(100)
            status_text.empty()
            progress_bar.empty()
            st.success('アップロードが完了しました。')
            
            #========================================
            # 「レイヤ一覧の表示」
            #========================================
            st.markdown('''
            <p class='my-text'>線画レイヤーの選択</p>''', unsafe_allow_html=True)
            progress_bar.empty()
            progress_bar_line = st.progress(0)
            status_text = st.empty()
            name_list = []
            img_list = []
            layer_number = 0
            #入力ファイルが変更されたかを判定
            if st.session_state.last_file_hash2 != current_file_hash:
                name_list = []
                img_list = []
                layer_number = 0
                
                # レイヤー処理を最適化
                visible_layers = [layer for layer in psd.descendants() 
                                 if not layer.is_group() and not layer.clipping_layer]
                
                total_layers = len(visible_layers)
                
                for idx, layer in enumerate(visible_layers):
                    layer.visible = True
                    base_img = layer.composite()
                    if base_img is not None:
                        name_list.append(layer.name)
                        # リサイズ処理を最適化
                        resized_img = base_img.resize((150, 150), Image.Resampling.NEAREST)
                        img_list.append(resized_img)
                        layer_number += 1
                    
                    # プログレスバーを更新
                    progress = int(((idx + 1) / total_layers) * 100)
                    progress_bar_line.progress(progress)
                    
                    status_text.text(f'レイヤー処理中... ({idx + 1}/{total_layers})')
                status_text.text('レイヤー処理が完了しました。線画レイヤーを選択してください。')
                progress_bar_line.empty()
                
                st.session_state.name_list = name_list
                st.session_state.img_list = img_list
                st.session_state.layer_number = layer_number
                st.session_state.last_file_hash2 = current_file_hash
            else:
                #ステートを利用（レイヤー一覧表示用）
                name_list = st.session_state.name_list
                img_list = st.session_state.img_list
                layer_number = st.session_state.layer_number
            #レイヤー一覧の表示
            cols = st.columns(10)
            selected_layers = []

            if "selected_layers" not in st.session_state:
                st.session_state.selected_layers = []

            # レイヤー名の重複をチェックして一意のIDを作成
            unique_layer_ids = []
            layer_name_count = {}
            
            for i, layer_name in enumerate(name_list):
                if layer_name in layer_name_count:
                    layer_name_count[layer_name] += 1
                    unique_id = f"{layer_name}#{layer_name_count[layer_name]}"
                else:
                    layer_name_count[layer_name] = 0
                    unique_id = f"{layer_name}#0"
                unique_layer_ids.append(unique_id)

            for i, img in enumerate(img_list):
                col = cols[i % 10]
                with col:
                    # 重複がある場合は番号付きで表示
                    display_name = name_list[i]
                    if layer_name_count[name_list[i]] > 0:
                        display_name = f"{name_list[i]}"
                    
                    st.image(img, caption=display_name, use_container_width=True)

                    checked = st.checkbox("線画選択", key=f"layer_chk_{i}")
                    unique_layer_id = unique_layer_ids[i]

                    if checked and unique_layer_id not in st.session_state.selected_layers:
                        st.session_state.selected_layers.append(unique_layer_id)
                    elif not checked and unique_layer_id in st.session_state.selected_layers:
                        st.session_state.selected_layers.remove(unique_layer_id)

            st.markdown("### 選択された線画レイヤー")
            if st.session_state.selected_layers:
                selected_cols = st.columns(10)
                for idx, unique_layer_id in enumerate(st.session_state.selected_layers):
                    try:
                        # unique_layer_idsからインデックスを取得
                        layer_idx = unique_layer_ids.index(unique_layer_id)
                        layer_name = name_list[layer_idx]
                        
                        # 表示名を作成
                        display_name = layer_name
                        if layer_name_count[layer_name] > 0:
                            display_name = f"{layer_name}" 
                        with selected_cols[idx % 5]:
                            st.image(img_list[layer_idx], caption=display_name, use_container_width=True)
                    except ValueError:
                        # レイヤーが見つからない場合はスキップ
                        continue
            else:
                st.info("まだレイヤーが選択されていません。")
            selected_layers = st.session_state.selected_layers

            if "line_img" not in st.session_state:
                st.session_state.line_img = None
            #レイヤーの選択
            line_img = Image.new("RGBA", psd_composite.size, (255, 255, 255, 0))
            
            if st.session_state.selected_layers:
                # 選択されたレイヤーの合成
                for unique_layer_id in st.session_state.selected_layers:
                    try:
                        # unique_layer_idsからインデックスを取得
                        layer_idx = unique_layer_ids.index(unique_layer_id)
                        original_layer_name = name_list[layer_idx]
                        
                        # PSDから対応するレイヤーを検索（インデックスベース）
                        layer_count = 0
                        target_index = int(unique_layer_id.split('#')[1])
                        
                        for layer in psd.descendants():
                            if layer.name == original_layer_name and not layer.is_group():
                                if layer_count == target_index:
                                    layer_img = layer.composite()
                                    if layer_img:
                                        line_img.paste(layer_img, (layer.left, layer.top), layer_img)
                                    break
                                layer_count += 1
                    except (ValueError, IndexError):
                        continue
                        
                line_resized = line_img.resize((256, 256), Image.LANCZOS)
                st.markdown('''<p class='box'>選択された線画レイヤー画像</p>''', unsafe_allow_html=True)
                st.image(line_resized)
            else:
                st.error('線画レイヤーが選択されていません。')
            
            #========================================
            # 「パラメータの調整」と「出力」
            #========================================
            if functions:
                #出力結果ダウンロード用のzipファイルを作成
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for function in functions:
                        #指定カラー表示システム
                        if function == 'Colors_Display_System':
                            st.markdown('''
                            <p class='my-text'>指定カラー表示システム</p>
                            <p class='box'>表示するカラーを選択</p>''', unsafe_allow_html=True)
                            if 'color_count' not in st.session_state:
                                st.session_state.color_count = 1
                            if st.button('カラーを増やす'):
                                st.session_state.color_count += 1
                            if st.button('カラーを減らす') and st.session_state.color_count > 1:
                                st.session_state.color_count -= 1
                            colors = []
                            thresholds = []
                            for i in range(st.session_state.color_count):
                                color = st.color_picker(f'##### カラー{i+1}の設定', '#000000')
                                colors.append(color)
                                st.write('###### 以下から色の許容範囲（±）を指定できます。')
                                threshold_R = st.slider(f"R{i+1}", 0, 30, key=f"R_slider_{i}")
                                threshold_G = st.slider(f"G{i+1}", 0, 30, key=f"G_slider_{i}")
                                threshold_B = st.slider(f"B{i+1}", 0, 30, key=f"B_slider_{i}")
                                thresholds.append((threshold_R, threshold_G, threshold_B))
                            st.markdown('''<p class='box'>各素材の表示切替</p>''', unsafe_allow_html=True)
                            background_display_switching1 = st.radio('###### 背景色の表示・非表示を切り替えられます。（非表示の場合、背景は透過されます。）', ('表示', '非表示'), key='back1_display', horizontal=True)
                            st.markdown('''<p class='box'>各素材のカラーを選択</p>''', unsafe_allow_html=True)
                            background1 = st.color_picker('###### 出力画像の背景カラーが変化します。', '#FFFFFF', key='back1')
                            #機能の実行
                            result_img1 = Colors_Display_System(psd, colors, thresholds, background_display_switching1, background1)
                            st.markdown('''<p class='box2'>指定カラー表示システムの出力結果</p>''', unsafe_allow_html=True)
                            #出力画像の表示
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(psd_composite, caption='元イラスト', use_container_width=True)
                            with col2:
                                st.image(result_img1, caption='出力結果', use_container_width=True)
                            result_buf1 = BytesIO()
                            result_img1.save(result_buf1, format='PNG')
                            result_buf1.seek(0)
                            #zipファイルに追加
                            zip_file.writestr('colorDisplaySystem_result.png', result_buf1.getvalue())
                    
                        #指定レイヤー輪郭表示システム
                        elif function == 'Layers_Contour_System':
                            st.markdown('''
                            <p class='my-text'>指定レイヤー輪郭表示システム</p>
                            <p class='box'>各値の調整</p>''', unsafe_allow_html=True)
                            threshold1 = st.slider('###### 二値化の閾値を調整できます。',0,255,128)
                            contour_size = st.slider('###### 輪郭線の幅を調整できます。',1,10,3)
                            st.markdown('''<p class='box'>各素材の表示切替</p>''', unsafe_allow_html=True)
                            layer_display_switching = st.radio('###### レイヤーの表示・非表示を切り替えられます。', ('表示', '非表示'), horizontal=True)
                            background_display_switching2 = st.radio('###### 背景色の表示・非表示を切り替えられます。（非表示の場合、背景は透過されます。）', ('表示', '非表示'), key='back2_display', horizontal=True)
                            st.markdown('''<p class='box'>各素材のカラーを選択</p>''', unsafe_allow_html=True)
                            color1 = st.color_picker('###### 輪郭線のカラーが変化します。', '#00FF00')
                            background2 = st.color_picker('###### 出力画像の背景カラーが変化します。', '#FFFFFF', key='back2')
                            st.markdown('''<p class='box'>表示する素材の選択</p>''', unsafe_allow_html=True)
                            #セッション変数の初期化（表示レイヤーの選択用）
                            if "selected_contours" not in st.session_state:
                                st.session_state.selected_contours = []
                            if "layers" not in st.session_state:
                                st.session_state.layers = None
                            selected_contours = st.multiselect(
                                '###### ～複数選択可能～',
                                name_list, 
                            )
                            st.write('###### :red[＊指定がない場合、線画レイヤーが自動で選択されます。]')
                            layers = Image.new("RGBA", psd_composite.size, (255, 255, 255, 0))
                            #選択レイヤーリストの合成
                            if not selected_contours:
                                layers = line_img
                            else:
                                #選択レイヤーが変更されていないかを確認
                                if selected_contours == st.session_state.selected_contours:
                                    layers = st.session_state.layers
                                else:
                                    for layer in psd.descendants():
                                        for item in selected_contours:
                                            if layer.name == item and not layer.is_group():
                                                layer_img = layer.composite()
                                                layers.paste(layer_img, (layer.left, layer.top), layer_img)
                                    #セッション変数を更新（表示レイヤーの選択用）
                                    st.session_state.selected_contours = selected_contours
                                    st.session_state.layers = layers
                            layers_resized = layers.resize((256, 256), Image.LANCZOS)
                            st.markdown('''<p class='box'>選択されたレイヤ</p>''', unsafe_allow_html=True)
                            st.image(layers_resized)
                            #機能の実行
                            result_img2 = Layers_Contour_System(psd, layers, threshold1, color1, contour_size, layer_display_switching, background_display_switching2, background2)
                            st.markdown("""<p class='box2'>指定レイヤー輪郭表示システムの出力結果</p>""", unsafe_allow_html=True)
                            #出力画像の表示
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(psd_composite, caption='元イラスト', use_container_width=True)
                            with col2:
                                st.image(result_img2, caption='出力結果', use_container_width=True)
                            result_buf2 = BytesIO()
                            result_img2.save(result_buf2, format='PNG')
                            result_buf2.seek(0)
                            #zipファイルに追加
                            zip_file.writestr('layerDisplaySystem_result.png', result_buf2.getvalue())
                            
                        #塗り漏れ検知システム
                        elif function == 'MissingPaint_Detection_System':
                            st.markdown('''
                            <p class='my-text'>塗り漏れ検知システム</p> 
                            <p class='box'>各値の調整</p>''', unsafe_allow_html=True)
                            reload = False
                            tolerance = st.slider('###### 「塗り漏れ」と判定する色の許容誤差を調整できます。',0,50,5)
                            circle_radius = st.slider('###### 「塗り漏れ」判定箇所の円の半径を調整できます。',10,100,50)
                            max_region_size = st.slider('###### 「塗り漏れ」と判定する領域のサイズを調整できます。',100,1000,500)
                            st.markdown('''<p class='box'>各素材の色を選択</p>''', unsafe_allow_html=True)
                            color2 = st.color_picker('###### 「塗り漏れ」検出箇所の色が変化します。', '#FF00FF')

                            st.caption('###### 出力結果が上手く表示されない場合には、以下の再実行ボタンを押してください。(再出力に数秒かかります)')
                            reload = st.button('###### 再実行', key='reload_missing')
                            #機能の実行
                            result_img3 = MissingPaint_Detection_System(psd, color2, tolerance, circle_radius, max_region_size, reload)
                            st.markdown('''<p class='box2'>塗り漏れ検知システムの出力結果</p>''', unsafe_allow_html=True)
                            #出力画像の表示
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(psd_composite, caption='元イラスト', use_container_width=True)
                            with col2:
                                st.image(result_img3, caption='出力結果', use_container_width=True)
                            result_buf3 = BytesIO()
                            result_img3.save(result_buf3, format='PNG')
                            result_buf3.seek(0)
                            #zipファイルに追加
                            zip_file.writestr('missingPaintDetectionSystem_result.png', result_buf3.getvalue())
                        
                        #消し忘れ検知システム
                        elif function == 'LineDrawingMistake_Detection_System':
                            st.markdown('''
                            <p class='my-text'>消し忘れ検知システム</p>
                            <p class='box'>各値の調整</p>''', unsafe_allow_html=True)
                            padding = st.slider('###### 検知範囲の拡張幅()', 
                                                1,10,3)
                            mask_size = st.slider('###### 検知対象の最大サイズ',10,1000,500, step=10)
                            gaussian_blur = st.radio('###### ガウシアンブラーを適用する(精度向上が見込めますが、見落としも発生しやすくなります)', 
                                                     options=['ON', 'OFF'], index=1, horizontal=True)

                            result_img4 = LineDrawingMistake_Detection_System(psd, line_img, 
                                                                              padding, mask_size, 
                                                                              gaussian_blur)
                            st.markdown('''<p class='box2'>消し忘れ検知システムの出力結果</p>''', unsafe_allow_html=True)
                            #出力画像の表示
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(psd_composite, caption='元イラスト', use_container_width=True)
                            with col2:
                                st.image(result_img4, caption='出力結果', use_container_width=True)
                            result_buf4 = BytesIO()
                            result_img4.save(result_buf4, format='PNG')
                            result_buf4.seek(0)
                            #zipファイルに追加
                            zip_file.writestr('missingLinesDetectionSystem_result.png', result_buf4.getvalue())      
                zip_buffer.seek(0)
                #ダウンロードボタンの表示
                st.download_button(
                    label="出力画像のダウンロードはこちらから",
                    data=zip_buffer, 
                    file_name="output_results.zip",
                    mime="application/zip",
                    type="primary"
                )
            else:
                st.error('機能が選択されていません。')