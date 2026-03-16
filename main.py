import cv2
import numpy as np
import io
import streamlit as st

from ultralytics import YOLO
from model_emnist import ModelEmnist
from torch import inference_mode
from torch import softmax

yolo_file = 'model_seg.pt'
emnist_file = 'model_emnist.pt'
map_emnist = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']

st.set_page_config(page_title = 'CV Pipeline Analysis', layout = 'wide')

st.markdown('''
<style>
    /* 1. Ограничиваем высоту всех изображений, чтобы они влезали в экран без прокрутки (70% от высоты экрана) */
    [data-testid='stImage'] img {
        max-height: 70vh;
        width: auto;
        object-fit: contain;
        margin-left: auto;
        margin-right: auto;
    }
    
    /* 2. Красим последнюю вкладку (Итог) в акцентный цвет */
    button[data-baseweb='tab']:last-child {
        background-color: #FF4B4B;
        border-radius: 5px 5px 0px 0px;
        padding: 0 15px;
    }
    button[data-baseweb='tab']:last-child p {
        color: white !important;
        font-weight: bold;
    }
</style>
''', unsafe_allow_html = True)

@st.cache_resource
def load_models():
    model = YOLO(yolo_file)
    model_emnist = ModelEmnist(emnist_file)
    model_emnist.eval()
    return model, model_emnist

try:
    model_yolo, model_emnist = load_models()
except Exception as e:
    st.error(f'Ошибка загрузки моделей. Проверьте пути к весам.\n{e}')
    st.stop()


def apply_preprocess(img: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(5, 5))
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)

    l_blur = cv2.medianBlur(l, 3)
    l = cv2.addWeighted(l, 2.5, l_blur, -1.5, 0)
    l = clahe.apply(l)
    l = cv2.GaussianBlur(l, (5, 5), 1)
    l = clahe.apply(l)
    l = cv2.medianBlur(l, 3)
    l = clahe.apply(l)

    img_res = cv2.merge((l, a, b))
    return cv2.cvtColor(img_res, cv2.COLOR_LAB2BGR)

def filtermask(img : np.ndarray, mask : np.ndarray) -> bool:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False
    contour = max(cnts, key = cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if perimeter == 0:
        return False

    k1 = (4 * np.pi * area) / (perimeter * perimeter)
    k2 = (img.mean(axis = 2) * mask).sum() / mask.sum()

    k1 = (k1 - 0.8) * 7
    k2 = k2 / 160

    if k1 + k2 < 0.45:
        return False
    return True

def apply_yolo(model : YOLO, img : np.ndarray):
    return model.predict(img, conf = 0.01, verbose = False, retina_masks = True)[0]

def apply_filter(img : np.ndarray, results):
    boxes = results.boxes.data.cpu().numpy()
    masks = results.masks.data.cpu().numpy()

    h, w = img.shape[:2]
    final_detections =[]
    sorted_indices = np.argsort(boxes[:, 4])[::-1]
    occupied_pixels = np.zeros((h, w), dtype = np.uint8)

    for idx in sorted_indices:
        conf = boxes[idx, 4]
        cls = int(boxes[idx, 5])

        mask = masks[idx]
        mask_resized = cv2.resize(mask, (w, h), interpolation = cv2.INTER_LINEAR)
        mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255

        if not filtermask(img, mask_binary):
            continue

        intersection = cv2.bitwise_and(mask_binary, occupied_pixels)
        if np.sum(intersection > 0) > np.sum(mask_binary > 0) * 0.3:
            continue

        final_detections.append({
            'mask': mask_binary,
            'class': cls,
            'conf': conf
        })

        occupied_pixels = cv2.bitwise_or(occupied_pixels, mask_binary)

    return final_detections

def apply_classification(img : np.ndarray, final_detections, model_emnist_instance):
    h, w = img.shape[:2]
    display_img = img.copy()
    counts = {0: 0, 1: 0, 2: 0}
    colors = {0: (255, 0, 0), 1: (0, 0, 255), 2: (0, 255, 255)}

    emnist_img = np.zeros((28, 28), dtype = np.float32)

    hue = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 0]
    lab_l = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0]
    lab_mask = (lab_l > 10) & (lab_l < 150)

    hue_data = []
    for det in final_detections:
        if det['class'] == 1:
            good_pixels = det['mask'] & lab_mask
            if good_pixels.sum() > 0:
                hue_data.append(((hue * good_pixels).sum() / good_pixels.sum()).item())
                det['hue'] = hue_data[-1]
            else:
                det['hue'] = 0

    if len(hue_data) > 1:
        hue_data = list(sorted(hue_data))
        max_gap = np.argmax([hue_data[i + 1] - hue_data[i] for i in range(len(hue_data) - 1)])
        red_yellow_split = (hue_data[max_gap + 1] + hue_data[max_gap]) * 0.5
    else:
        red_yellow_split = 0

    for det in final_detections:
        mask = det['mask']
        cls = det['class']

        if cls == 1 and det.get('hue', 0) < red_yellow_split:
            cls += 1

        color = colors.get(cls, (255, 255, 0))
        counts[cls] += 1

        mask_boolean = mask > 0
        display_img[mask_boolean] = display_img[mask_boolean] * 0.5 + np.array(color) * 0.5

        M = cv2.moments(mask)
        if M['m00'] != 0:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])

            eX = int(np.round(((cX / (w / 2)) - 1) * 10 + 14))
            eY = int(np.round(((cY / (h / 2)) - 1) * 10 + 14))
            
            eX, eY = max(0, min(27, eX)), max(0, min(27, eY))
            cv2.circle(emnist_img, (eX, eY), 1, 1, -1, lineType = cv2.LINE_AA)

            label = str(counts[cls])
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display_img, (cX-5, cY-th-5), (cX+tw+5, cY+5), (0,0,0), -1)
            cv2.putText(display_img, label, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    info_text = f'Eggman: {counts[0]} | Red: {counts[1]} | Yellow: {counts[2]}'

    kernel = np.ones((2,2), np.float32)
    emnist_img = cv2.dilate(emnist_img, kernel, iterations=1)
    emnist_img = cv2.GaussianBlur(emnist_img, (3, 3), 0)
    cv2.imwrite('emnist_real.png', emnist_img * 255)
    emnist_img = (emnist_img - 0.5) / 0.5

    with inference_mode():
        pred = model_emnist_instance(emnist_img.T[None, None])[0]
        if pred.max() > -2:
            info_text += f' | Char: {map_emnist[pred.argmax()]} ({softmax(pred, -1).max().item() * 100:.0f}%)'

    cv2.rectangle(display_img, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(display_img, info_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return display_img, counts, info_text

def process_image(img_bgr):
    stages = {}
    stages['Оригинал'] = img_bgr.copy()
    
    preprocessed_img = apply_preprocess(img_bgr.copy())
    stages['Предобработка'] = preprocessed_img

    results = apply_yolo(model_yolo, preprocessed_img)
    if not results.masks:
        return None, 'Объекты не найдены'

    stages['YOLO (сырая)'] = results.plot()

    final_detections = apply_filter(img_bgr, results)
    
    filtered_img = img_bgr.copy()
    for det in final_detections:
        mask_bool = det['mask'] > 0
        filtered_img[mask_bool] = filtered_img[mask_bool] * 0.5 + np.array([0, 255, 0]) * 0.5
    stages['После фильтрации'] = filtered_img

    display_img, counts, info_text = apply_classification(img_bgr.copy(), final_detections, model_emnist)
    stages['Итог'] = display_img
    
    return stages, info_text


st.title('👁️ Cегментация и распознавание Матвея')

uploaded_file = st.file_uploader('Загрузите изображение (JPG, PNG)', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    with st.spinner('Обработка изображения...'):
        stages, info = process_image(img_bgr)
        
    if stages is None:
        st.warning(info)
    else:
        st.success(info)

        tab_all, tab_1, tab_2, tab_3, tab_4, tab_5 = st.tabs([
            'Все вместе', 'Оригинал', 'Предобработка', 'Сырая YOLO', 'После фильтрации', '👁️ Итог 👁️'
        ])

        def show_img(img, caption=''):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption=caption, width='stretch')

        with tab_all:
            col1, col2, col3 = st.columns(3)
            with col1: show_img(stages['Оригинал'], '1. Оригинал')
            with col2: show_img(stages['Предобработка'], '2. Предобработка')
            with col3: show_img(stages['YOLO (сырая)'], '3. YOLO (сырая)')
            
            col4, col5, col6 = st.columns(3)
            with col4: show_img(stages['После фильтрации'], '4. После фильтрации')
            with col5: show_img(stages['Итог'], '5. Итог')
            
        with tab_1: show_img(stages['Оригинал'])
        with tab_2: show_img(stages['Предобработка'])
        with tab_3: show_img(stages['YOLO (сырая)'])
        with tab_4: show_img(stages['После фильтрации'])
        with tab_5: show_img(stages['Итог'])

        st.markdown('---')

        result_img = stages['Итог']
        is_success, buffer = cv2.imencode('.jpg', result_img)
        if is_success:
            io_buf = io.BytesIO(buffer)
            
            st.download_button(
                label='Скачать результат',
                data=io_buf,
                file_name=f'result_{uploaded_file.name}',
                mime='image/jpeg',
                type='primary' 
            )
