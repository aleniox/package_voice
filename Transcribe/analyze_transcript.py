import re
import subprocess
import time
import torch
import whisperx
from langchain_groq import ChatGroq
from pyannote.audio import Pipeline
import librosa
from PySide6.QtWidgets import *
from PySide6.QtGui import *
from Transcribe.config import OUTPUT_FOLDER, text_fail, SPEAKER_FOLDER, NEON_RED, NEON_GREEN, RESET_COLOR, NEON_CYAN
from threading import Thread
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
import queue
from huggingface_hub import hf_hub_download
import os
import secrets
import gc
from dotenv import load_dotenv
import json
from pyvi import ViTokenizer, ViPosTagger
from pydub import AudioSegment
import random

load_dotenv()


def check_words_in_text(words, text):
    """
    Kiểm tra xem các từ trong danh sách có trong đoạn văn không.
    :param words: Danh sách các từ cần kiểm tra.
    :param text: Đoạn văn cần kiểm tra.
    :return: Một danh sách chứa các từ có trong đoạn văn.
    """
    found_words = [word for word in words if word in text]
    for para in found_words:
        text = text.replace(para, "")
    text = " ".join(text.split())
    if len(text) > 2:
        return found_words, text
    else:
        return None, None


def check_format_text(text):
    # Xóa dấu câu không cần thiết giữa các từ
    if text and text[-1] not in ['.', '!', '?']:
        text += '.'
    text = re.sub(r'([,.!?])([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'(\s+)', ' ', text)

    text = re.sub(r'([,.!?])\s*([,.!?])', r'\1', text)
    text = re.sub(r'\s*([,.!?])', r'\1', text)
    text = re.sub(r'([,.!?])\s+', r'\1 ', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1. \2', text)
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    sentences = [s.capitalize() for s in sentences]
    return ' '.join(sentences)


class ReformatSentence(QObject):
    update_text = Signal(str)

    def __init__(self, paragraph):
        super().__init__()

        self.query_prompt = PromptTemplate(
            template="""Hãy sửa các lỗi chính tả và ngữ pháp không đúng trong câu sau như đặt dấu câu sao cho đúng vị trí không được thay đổi nội dung của câu: "{sentence}". Trả về JSON với khóa 'output' chứa câu đã được sửa.""",
            # template=""""Sửa lỗi chính tả và ngữ pháp của câu tiếng việt sau sao cho phù hợp "{sentence}". Trả về câu đã được sửa.""",
            input_variables=["sentence"],
        )
        self.paragraph = paragraph

    def run(self):
        try:
            print("Đang load model đợi chút!")
            # logging.info("Đang load model đợi chút!")
            try:
                self.llm = ChatGroq(temperature=0, model_name="llama-3.1-70b-versatile",
                                    api_key=os.getenv("GROQ_API_KEY"))
                self.generate_chain = self.query_prompt | self.llm | JsonOutputParser()
                text = self.generate_chain.invoke(
                    {'sentence': self.paragraph})["output"]
                print("Sử dụng model v1")
            except Exception as e:
                from llama_cpp import Llama
                self.llm = Llama(model_path="model/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                                 chat_format="chatml",
                                 n_gpu_layers=32,
                                 n_ctx=2048)
                output = self.llm.create_chat_completion(
                    messages=[
                        {
                            "role": "system",
                            "content": "Bạn là AI dùng để sửa lỗi chính tả và ngữ pháp mà không làm thay đổi nội dung câu.",
                        },
                        {"role": "user", "content": f"Hãy sửa các lỗi chính tả và ngữ pháp không đúng trong câu sau như đặt dấu câu sao cho đúng vị trí không được thay đổi nội dung của câu: '{self.paragraph}'. Trả về JSON với khóa 'output' chứa câu được viết lại có chính tả và ngữ pháp đúng."},
                    ],
                    response_format={
                        "type": "json_object",
                    },
                    temperature=0.7,
                    # stream=True,
                )
                print("Sử dụng model v1 local")
                text = output['choices'][0]['message']['content']
                text = json.loads(text)["output"]
        except Exception as e:
            text = self.paragraph
        self.update_text.emit(text)
        self.emtry_memory()

    def emtry_memory(self):
        del self.llm
        del self.generate_chain
        gc.collect()
        torch.cuda.empty_cache()


class AnalyzeTranscript(Thread):
    def __init__(self, cluster_select: bool = True, wav_file: str = None, select_language: str = None, device: str = "cuda", use_auth_token: str = None):
        super().__init__()
        self.wav_file = wav_file
        self.cluster_option_select = cluster_select
        self.select_language = select_language
        self.device = device
        self.classify = 0
        self.use_auth_token = use_auth_token
        if "cuda" in device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.compute_type = "float16"
        else:
            self.device = "cpu"
            self.compute_type = "int8"
        self.check_language = True
        self.kwargs = {"min_speakers": 1, "max_speakers": 10}
        self.generate_chain = None
        self.result = queue.Queue()

        print(f'Đang dùng trên: {self.device}')

    def init_model(self):
        try:
            self.transcript_model = whisperx.load_model(
                "large-v3", "cuda" if "cuda" in self.device else "cpu", compute_type=self.compute_type)
        except Exception as e:
            print(f"Error load model {e}")
            self.transcript_model = whisperx.load_model(
                "large-v3", "cuda" if "cuda" in self.device else "cpu", compute_type=self.compute_type)

    def classify_from_audio(self, audio_path):
        if self.use_auth_token is not None:
            if "hf_" not in self.use_auth_token:
                self.use_auth_token = os.getenv("HUGGING_FACE_KEY")
        else:
            self.use_auth_token = os.getenv("HUGGING_FACE_KEY")

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=self.use_auth_token)
        pipeline.to(torch.device(self.device))
        print("Đang tách giọng nói...")
        start = time.time()
        vad = pipeline(audio_path, **self.kwargs)
        print(
            "----------" + NEON_GREEN + "Thời gian tách giọng nói dùng " + NEON_RED + f"{self.device}: " + NEON_CYAN + f"{(time.time() - start)/60} phút"+RESET_COLOR + "----------")

        self.total_time_classify = f"- Thời gian tách giọng nói: {time.time() - start:.2f}s"
        gc.collect()
        del pipeline
        torch.cuda.empty_cache()
        return vad

    def cut_audio_to_chunks(self, vad, audio_path):
        audio = AudioSegment.from_file(audio_path, format="wav")
        speaker_color = {}
        data_segment = []
        speaker_point = "@@@"
        start_ms = 0
        end_ms = 0
        name_folder = self.wav_file.split(
            "/")[-1].replace(".wav", "")+"_"+str(secrets.token_hex(9))

        for turn, _, speaker in vad.itertracks(yield_label=True):
            if speaker not in speaker_color:
                speaker_color[speaker] = "#{:06x}".format(
                    random.randint(0, 0xFFFFFF))
                os.makedirs(
                    f"{SPEAKER_FOLDER}/{name_folder}/{speaker}", exist_ok=True)
            if speaker_point != speaker:
                if end_ms - start_ms > 00:
                    path_audio_segment = f"{SPEAKER_FOLDER}/{name_folder}/{speaker_point}/speaker_{start_ms}_to_{end_ms}.wav"
                    data_segment.append({
                        "speaker": speaker_point,
                        "start": start_ms,
                        "end": end_ms,
                        "path": path_audio_segment,
                        "sample_rate": audio.frame_rate,
                        "color": speaker_color[speaker_point],
                    })
                    segment = audio[start_ms:end_ms]
                    segment.export(path_audio_segment, format="wav")

                start_ms = int(turn.start * 1000)
                speaker_point = speaker

            end_ms = int(turn.end * 1000)

        if end_ms - start_ms > 00:
            path_audio_segment = f"{SPEAKER_FOLDER}/{name_folder}/{speaker_point}/speaker_{start_ms}_to_{end_ms}.wav"
            data_segment.append({
                "speaker": speaker_point,
                "start": start_ms,
                "end": end_ms,
                "path": path_audio_segment,
                "sample_rate": audio.frame_rate,
                "color": speaker_color[speaker_point],
            })
            segment = audio[start_ms:end_ms]
            segment.export(path_audio_segment, format="wav")
        print(f"Có {len(speaker_color)} người nói trong audio!")

        return data_segment

    def transcribe(self, audio_file, batch_size):
        audio = whisperx.load_audio(audio_file)
        text = []
        result = self.transcript_model.transcribe(
            audio, batch_size=batch_size, language=self.select_language)
        for segment in result["segments"]:
            text.append(segment["text"])
        return ".".join(text), result["language"]

    def speech_to_text(self, segment):
        for index, chunk in enumerate(segment):
            start = time.time()
            output = self.transcribe(chunk['path'], 8)
            try:
                text, language = output["text"]
            except:
                text, language = output
                # language = 'vi'
            _, text = check_words_in_text(text_fail, text)
            if text == None:
                continue
            text = check_format_text(text)
            # list_word = text.split()
            List_all = ViPosTagger.postagging(ViTokenizer.tokenize(text))
            list_word = [word.replace("_", " ") for word in List_all[0]]
            end = time.time()
            self.result.put({
                "speaker": chunk["speaker"],
                "text": text,
                "start": chunk["start"],
                "end": chunk["end"],
                "path": chunk["path"],
                "language": language,
                "sample_rate": chunk["sample_rate"],
                "text": list_word
            })

            print(
                f"Thời gian chạy đoạn từ {chunk['start']/1000} đến {chunk['end']/1000}: {end-start}s", )

    def convert_audio(self, input_file):
        if ".wav" not in input_file:
            format_audio = "."+input_file.split(".")[-1:][0]
            output_file = input_file.replace(format_audio, ".wav")
            command = ['ffmpeg', '-y', '-i', input_file, output_file]
            subprocess.run(command)
            return output_file
        return input_file

    def transcribe_audio(self, input_file):
        samples, sample_rate = librosa.load(input_file, sr=None)
        duration = librosa.get_duration(y=samples, sr=sample_rate)
        if duration < 1000:
            print("-----------Audio ngắn-----------")
            self.init_model()
            list_segments, input_file = self.transcribe_audio_short(
                input_file, sample_rate)
        else:
            print("-----------Audio dài-----------")
            # logging.info("-----------Audio dài-----------")
            vad = self.classify_from_audio(input_file)
            self.init_model()
            self.transcribe_audio_long(
                self.cut_audio_not_classify(vad, input_file))

    def transcribe_audio_short(self, input_file, sample_rate):
        from pydub import AudioSegment
        list_segments = []
        output = self.transcript_model.transcribe(input_file)
        audio = AudioSegment.from_file(input_file, format="wav")
        language = output["language"]
        segments = output['segments']
        name_folder = input_file.split("/")[-1].replace(".wav", "")
        if not os.path.exists(f"{OUTPUT_FOLDER}/{name_folder}/audio_classify"):
            os.makedirs(
                f"{OUTPUT_FOLDER}/{name_folder}/audio_classify", exist_ok=True)
        for segment in segments:
            path_audio_segment = f"{OUTPUT_FOLDER}/{name_folder}/audio_classify/speaker_{segment['start']}_to_{segment['end']}.wav"

            list_segments.append({
                "speaker": "",
                "text": segment['text'],
                "start": float(segment['start']),
                "end": float(segment['end']),
                "sample_rate": sample_rate,
                "color": "#{:06x}".format(
                    random.randint(0, 0xFFFFFF)),
                "language": language,
                "path": path_audio_segment,
            })

            segment = audio[int(segment['start']*1000)
                                :int(segment['end']*1000)]
            segment.export(path_audio_segment, format="wav")
        return list_segments, input_file

    def cut_audio_not_classify(self, vad, audio_path):
        i = 0
        data_segment = []
        start_point = 0
        end_point = 0
        audio = AudioSegment.from_file(audio_path, format="wav")
        name_folder = self.wav_file.split(
            "/")[-1].replace(".wav", "")+"_"+str(secrets.token_hex(9))
        os.makedirs(
            f"{SPEAKER_FOLDER}/{name_folder}", exist_ok=True)
        for turn, _, speaker in vad.itertracks(yield_label=True):
            # print(turn.start, turn.end,  speaker)
            if i < 1:
                start_point = int(turn.start*1000)
            i += 1
            end_point = int(turn.end*1000)
            if i % 20 == 0 or end_point-start_point > 30000:
                path_audio_segment = f"{SPEAKER_FOLDER}/{name_folder}/speaker_{start_point}_to_{end_point}.wav"

                data_segment.append({
                    "speaker": f"SPEAKER_{i//4}",
                    "start": start_point,
                    "end": end_point,
                    "path": path_audio_segment,
                    "sample_rate": audio.frame_rate,
                    "color": "#{:06x}".format(random.randint(0, 0xFFFFFF))})
                segment = audio[start_point:end_point]
                segment.export(path_audio_segment, format="wav")
                start_point = int(turn.start*1000)

        return data_segment

    def transcribe_audio_long(self, segment):
        # pass
        total_ = 0
        for index, chunk in enumerate(segment):
            start = time.time()
            output, language = self.transcribe(chunk['path'], 8)
            _, text = check_words_in_text(text_fail, output)
            if text == None:
                continue
            text = check_format_text(text)
            end = time.time()
            total_ += end-start
            list_word = text.split()

            print(
                f"Thời gian chạy đoạn từ {chunk['start']/1000} đến {chunk['end']/1000}: {end-start}s", )
        print("----------"+NEON_GREEN +
              f"Tổng thời gian xử lý audio trên {self.device}: " + NEON_CYAN + f"{total_/60} phút" + RESET_COLOR + "----------")

    def run(self):

        print("Analyze transcript...")
        start_time = time.time()
        audio_path = self.convert_audio(self.wav_file)

        if self.cluster_option_select:
            self.classify = 1
            audio_segments = self.classify_from_audio(audio_path)
            self.init_model()
            segments = self.cut_audio_to_chunks(audio_segments, audio_path)
            self.speech_to_text(segments)
        else:
            self.transcribe_audio(audio_path)
        print("----------"+NEON_GREEN +
              f"Tổng thời gian phân tích trên "+NEON_RED+f"{self.device}: " + NEON_CYAN + f"{(time.time() - start_time)/60} phút" + RESET_COLOR + "----------")

        self.emtry_memmory()

    def generate_random_color(self):
        return "#{:06x}".format(random.randint(0, 0xFFFFFF))

    def emtry_memmory(self):
        gc.collect()
        # del self.generate_chain
        del self.transcript_model
        torch.cuda.empty_cache()

    def assign_color_to_speaker(self, speaker):
        if speaker not in self.speaker_colors:
            # Gán màu mới nếu loa chưa có màu
            self.speaker_colors[speaker] = self.generate_random_color()
        return self.speaker_colors[speaker]


class VoiceRestore(QObject):
    update_text = Signal(str)
    def __init__(self, checkpoint, input_audio, output_audio):
        pass
        

class RetriverDoc(QObject):
    update_text = Signal(str)

# if __name__ == "__main__":
