import onnxruntime as ort
import torch
import numpy as np
from torchvision.transforms import v2


# 모델을 로드합니다. 사용자 상황에 따라 GPU도 사용 가능하도록 설정했습니다.
ort_session = ort.InferenceSession(
    "src/assets/mnist-12-int8.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# streamlit의 canvas로 받는 이미지는 단순히 배열이어서 먼저 PIL 이미지로 변환해야 됩니다.
transform = v2.Compose([
    v2.ToPILImage(),
    v2.Grayscale(num_output_channels=1),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize(28) # 모델이 사용하는 크기인 28*28에 맞춰줍니다.
])

def predict_number(img):
    img = transform(img)

    # 이미지는 기본적으로 channels, height, width로 구성되는 3차원 이미지입니다.
    # 따라서, 모델이 쓸 수 있도록 unsqueeze하여 배치 크기 1로 만들어줍니다.
    ort_inputs = {"Input3": img.unsqueeze(0).numpy()}
    recon = ort_session.run(None, ort_inputs)[0][0]

    # 최종 출력은 numpy array입니다.
    # 그렇기에 torch의 softmax를 쓰는 대신 softmax을 사용하여 계산합니다. 
    return np.exp(recon)/sum(np.exp(recon))