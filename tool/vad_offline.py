from funasr import AutoModel

model = AutoModel(model="fsmn-vad", model_revision="v2.0.4")

wav_file = f"{model.model_path}/example/vad_example.wav"
res = model.generate(input=wav_file)
print(res)