import os
from funasr import AutoModel
# paraformer-zh is a multi-functional asr model
# use vad, punc, spk or not as you need
model = AutoModel(model="paraformer-zh", model_revision="v2.0.4",
                #   vad_model="fsmn-vad", vad_model_revision="v2.0.4",
                #   punc_model="ct-punc-c", punc_model_revision="v2.0.4",
                #   spk_model="cam++", spk_model_revision="v2.0.2",
                  disable_update=True
                  )

#得到当前文件的路径
current_path = os.path.dirname(os.path.abspath(__file__))
res = model.generate(input=f"{current_path}/sample/multi_speaker.wav", 
            batch_size_s=300, 
            hotword='魔搭')
print(res)