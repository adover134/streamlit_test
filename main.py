import os


# 모델을 아직 다운로드 받지 않았다면 다운로드를 받아야 합니다.
if not os.path.exists('src/assets/mnist-12-int8.onnx'):
    # github로부터 사용할 모델을 다운로드받아서, 실행시키기 위해 있어야 하는 위치에 배치합니다.
    os.system('wget https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/mnist/model/mnist-12-int8.onnx')
    os.system('mv mnist-12-int8.onnx src/assets/mnist-12-int8.onnx')
# 사용할 프로그램을 자동으로 실행합니다.
os.system('streamlit run app.py')