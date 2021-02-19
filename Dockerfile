FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-0
WORKDIR /root

COPY cuda_extras /usr/local/cuda-10.1/extras
# This next one is less likely to work:
COPY cuda_extras /usr/local/cuda-10.0/extras

COPY trainer ./trainer

ENTRYPOINT [ "python", "trainer/train_kerasNet.py" ]
