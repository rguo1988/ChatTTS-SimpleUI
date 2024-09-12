import streamlit as st
import ChatTTS
import torch
import soundfile
import numpy as np
import pyloudnorm as pyln
import random

# MODEL_PATH = "D:/ai/ChatTTS-main/asset"


# function defs
def RefineText(original_text, params_refine_text=None):
    st.session_state.refined_text = tts.infer(
        original_text.replace("\n", ""),
        params_refine_text=params_refine_text,
        refine_text_only=True,
    )


def StablizeTone(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ReplaceText(text_list):
    for text in text_list:
        text = text.replace("\n", "。")
        text = text.replace("！", "。")
        text = text.replace("“", "")
        text = text.replace("”", "")
        text = text.replace("...", "")
        text = text.replace("——", ",")
        text = text.replace("-", ",")
        text = text.replace("?", ",")
    return text_list


def CutTexts(string, num=120, including_space=False):
    texts_cutted = []
    cut_symbos = ["。", "，", "！", "？", "……", "…", "：", "；", "、"]
    if including_space:
        num *= 2
    while len(string) > num:
        string_temp = string[:num]
        cut_idx = 0
        for sym in cut_symbos:
            if cut_idx < string_temp.rfind(sym):
                cut_idx = string_temp.rfind(sym)
        texts_cutted.append(string[: cut_idx + 1])
        string = string[cut_idx + 1 :]
    texts_cutted.append(string)

    if len(texts_cutted) > 1:
        if len(texts_cutted[-1]) < 15:
            texts_cutted[-2] += texts_cutted[-1]
            del texts_cutted[-1]
    return texts_cutted


def NormalizeAudioVolume(wav):
    # peak_normalized_audio = pyln.normalize.peak(wav, -1.0)
    meter = pyln.Meter(24000)  # create BS.1770 meter
    wav_loudness = meter.integrated_loudness(wav[0])
    wav_norm = pyln.normalize.loudness(wav[0], wav_loudness, -12.0)
    return wav_norm


# initialize cache
@st.cache_resource
def load_chattts_model():
    tts = ChatTTS.Chat()
    tts.load(compile=False)
    return tts


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

tts = load_chattts_model()
speeds = ["[speed_1]", "[speed_2]", "[speed_3]", "[speed_4]", "[speed_5]"]
params_refine_text = ChatTTS.Chat.RefineTextParams(prompt="[oral_2][laugh_0][break_6]")
spk = tts._encode_spk_emb(
    torch.load(
        "./spk/seed_1453_restored_emb.pt",
        weights_only=True,
        map_location=torch.device(device),
    )
)
spk_seed = 1453

if "refined_text" not in st.session_state:
    st.session_state.refined_text = [""]
if "random_seed" not in st.session_state:
    st.session_state.random_seed = 1453

# sidebar
with st.sidebar:
    st.subheader("参数设置")

    temperature_slider = st.slider(
        "temperature", min_value=0.0, max_value=1.0, value=0.1
    )
    topp_slider = st.slider("top_P", min_value=0.1, max_value=0.9, value=0.7)
    topk_slider = st.slider("top_K", min_value=1, max_value=20, value=20)
    speed_slider = st.slider("speed", min_value=0, max_value=5, value=1)

    tone_selectbox = st.selectbox(
        "请选择音色：",
        [
            "青年男性1",
            "青年男性2",
            "中年男性",
            "青年女性1",
            "青年女性2",
            "中年女性",
            "自定义音色",
            "上传本地音色",
        ],
    )
    if tone_selectbox == "青年男性1":
        spk = tts._encode_spk_emb(
            torch.load(
                "./spk/seed_1453_restored_emb.pt",
                weights_only=True,
                map_location=torch.device(device),
            )
        )
        spk_seed = 1453
    if tone_selectbox == "青年男性2":
        spk = tts._encode_spk_emb(
            torch.load(
                "./spk/seed_1996_restored_emb.pt",
                weights_only=True,
                map_location=torch.device(device),
            )
        )
        spk_seed = 1996
    if tone_selectbox == "中年男性":
        spk = tts._encode_spk_emb(
            torch.load(
                "./spk/seed_715_restored_emb.pt",
                weights_only=True,
                map_location=torch.device(device),
            )
        )
        spk_seed = 715
    if tone_selectbox == "青年女性1":
        spk = tts._encode_spk_emb(
            torch.load(
                "./spk/seed_1397_restored_emb.pt",
                weights_only=True,
                map_location=torch.device(device),
            )
        )
        spk_seed = 1397
    if tone_selectbox == "青年女性2":
        spk = tts._encode_spk_emb(
            torch.load(
                "./spk/seed_1151_restored_emb.pt",
                weights_only=True,
                map_location=torch.device(device),
            )
        )
        spk_seed = 1151
    if tone_selectbox == "中年女性":
        spk = tts._encode_spk_emb(
            torch.load(
                "./spk/seed_395_restored_emb.pt",
                weights_only=True,
                map_location=torch.device(device),
            )
        )
        spk_seed = 395

    if tone_selectbox == "自定义音色":
        col_text, col_input, col_button = st.columns([0.6, 1.8, 1])
        with col_text:
            st.write("音色seed")
        with col_button:
            random_button = st.button("随机")
            torch.manual_seed(int(spk_seed))
            if random_button:
                st.session_state.random_seed = random.randint(1, 4294967295)
                spk_seed = st.session_state.random_seed
                torch.manual_seed(int(spk_seed))
        with col_input:
            spk_seed = st.text_input(
                label="音色seed",
                value=st.session_state.random_seed,
                label_visibility="collapsed",
            )

        spk = tts.sample_random_speaker()

    if tone_selectbox == "上传本地音色":
        uploaded_file = st.file_uploader("上传音色文件", type=["pt"])
        if uploaded_file is not None:
            spk = tts._encode_spk_emb(
                torch.load(
                    uploaded_file, weights_only=True, map_location=torch.device(device)
                )
            )

    long_text = st.checkbox("长文本自动拼接")
    if long_text:
        cut_length = st.number_input("分段长度", value=120)

# main area
if device == "cuda":
    st.title("ChatTTS-SimpleUI(启用GPU)")
else:
    st.title("ChatTTS-SimpleUI(未启用GPU)")
tab_direct, tab_refine = st.tabs(["直接生成语音", "口语化文本后生成语音"])
with tab_direct:
    text_d = st.text_area(
        "文本", placeholder="请输入原始文本", label_visibility="collapsed"
    )

    if long_text:
        text_d = CutTexts(text_d, cut_length)
    text_d = ReplaceText(text_d)

    generate_button_d = st.button("生成语音")
    if generate_button_d:
        params_infer_code = ChatTTS.Chat.InferCodeParams(
            prompt=speeds[speed_slider],
            temperature=temperature_slider,
            top_K=topk_slider,
            top_P=topp_slider,
            spk_emb=spk,
        )
        if long_text:
            wavs_d = np.array([])
            for text in text_d:
                StablizeTone(int(spk_seed))
                wav = tts.infer(
                    text.replace("\n", ""),
                    use_decoder=True,
                    params_refine_text=params_refine_text,
                    params_infer_code=params_infer_code,
                    skip_refine_text=True,
                )[0]

                # normalize audio volume
                wav_norm = NormalizeAudioVolume(wav)
                wavs_d = np.append(wavs_d, wav_norm)
            wavs_d = wavs_d.flatten()
        else:
            wav = tts.infer(
                text_d.replace("\n", ""),
                use_decoder=True,
                params_refine_text=params_refine_text,
                params_infer_code=params_infer_code,
                skip_refine_text=True,
            )[0]
            # normalize audio volume
            wavs_d = NormalizeAudioVolume(wav)

        col1, col2 = st.columns(2)
        with col1:
            st.audio(wavs_d, format="audio/wav", sample_rate=24000)
        with col2:
            try:
                soundfile.write("output.wav", wavs_d, 24000)
            except:
                pass
            st.download_button(
                label="保存音频",
                data="output.wav",
                file_name="output.wav",
                mime="audio/wav",
            )

with tab_refine:
    original_text = st.text_area("原始文本", placeholder="请输入原始文本")
    original_text = ReplaceText(original_text)
    refine_button = st.button("生成口语化文本(Refine Text)")

    if refine_button:
        RefineText(original_text, params_refine_text)

    refined_texts = st.text_area(
        "口语化文本(可自行添加口语化标签[uv_break]，[laugh]等)",
        value=st.session_state.refined_text[0],
    )

    generate_button_r = st.button("生成语音", key="generate_button_r")
    if generate_button_r:
        if long_text:
            refined_texts = CutTexts(refined_texts, cut_length, including_space=True)

        params_infer_code = ChatTTS.Chat.InferCodeParams(
            prompt=speeds[speed_slider],
            temperature=temperature_slider,
            top_K=topk_slider,
            top_P=topp_slider,
            spk_emb=spk,
        )
        if long_text:
            wavs_r = np.array([])
            for text in refined_texts:
                StablizeTone(int(spk_seed))
                wav = tts.infer(
                    text,
                    use_decoder=True,
                    params_refine_text=params_refine_text,
                    params_infer_code=params_infer_code,
                    skip_refine_text=True,
                )[0]

                # normalize audio volume
                wav_norm = NormalizeAudioVolume(wav)
                wavs_r = np.append(wavs_r, wav_norm)
            wavs_r = wavs_r.flatten()
        else:
            wav = tts.infer(
                refined_texts,
                use_decoder=True,
                params_infer_code=params_infer_code,
                params_refine_text=params_refine_text,
                skip_refine_text=True,
            )[0]
            # normalize audio volume
            wavs_r = NormalizeAudioVolume(wav)
        col1, col2 = st.columns(2)
        with col1:
            st.audio(wavs_r, format="audio/wav", sample_rate=24000)
        with col2:
            try:
                soundfile.write("output.wav", wavs_r, 24000)
            except:
                pass
            st.download_button(
                label="保存音频",
                data="output.wav",
                file_name="output.wav",
                mime="audio/wav",
            )
