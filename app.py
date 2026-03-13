import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import pandas as pd
import torch
from torchvision.transforms import v2
from src.model import predict_number


st.set_page_config(page_title="MNIST 손글씨 예측하기", layout="wide")
st.title("MNIST 손글씨 숫자 예측하기")
st.write("캔버스에 그린 숫자를 모델이 예측합니다.")

# 캔버스 설정
stroke_width = st.sidebar.slider("Stroke width: ", 1, 100, 5) # 선 굵기를 조정하는 slider입니다.
stroke_color = "#FFFFFF" # MNIST 데이터셋을 고려하여, 선은 흰색으로 고정하였습니다.
bg_color = "#000" # 마찬가지 이유로 배경은 검은색으로 고정하였습니다.
drawing_mode = "freedraw" # 손글씨라는 특징을 살리기 위해 freedraw로 지정하였습니다.

canvas_result = st_canvas( # 캔버스 설정은 앞서 작성한 것들을 활용합니다.
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height = 100, # 숫자 하나를 그리기에 충분한 공간을 줬습니다.
    width = 100, # 정사각형임을 고려하여 높이와 같은 값을 줬습니다.
    drawing_mode=drawing_mode,
    key="canvas",
)

if canvas_result.image_data is not None:
    st.title('입력 이미지')
    st.text('모델이 실제로 입력받는 이미지로, 그린 것과 차이가 있을 수 있습니다.')
    transform = v2.Compose([
        v2.ToPILImage(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(28) # 모델이 사용하는 크기인 28*28에 맞춰줍니다.
    ])
    ti = transform(canvas_result.image_data)
    st.image(ti.permute(1, 2, 0).numpy(), clamp=True, channels='')
    res = predict_number(canvas_result.image_data)
    prediction=pd.DataFrame.from_dict({'result':res, 'nums':list(range(10))})
    st.title('예측 결과')
    st.text('모델이 예측한 각 숫자 별 확률입니다.')
    st.dataframe(prediction)
    st.bar_chart(prediction, y='result', x='nums')