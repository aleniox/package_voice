import os
import pickle
from huggingface_hub import hf_hub_download
# documents = documents_path = os.path.join(os.path.expanduser("~"), "Documents")
documents = os.getcwd()
STATIC_FOLDER = os.path.join(documents, "VOICEMASTER")
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)
OUTPUT_FOLDER = os.path.join(STATIC_FOLDER, "OUTPUT")
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

DOCUMENT_FOLDER = os.path.join(STATIC_FOLDER, "DOCUMENT")
if not os.path.exists(DOCUMENT_FOLDER):
    os.makedirs(DOCUMENT_FOLDER)

HASH_FOLDER = os.path.join(STATIC_FOLDER, "HASH")
if not os.path.exists(HASH_FOLDER):
    os.makedirs(HASH_FOLDER)

SPEAKER_FOLDER = os.path.join(STATIC_FOLDER, "SPEAKER")
if not os.path.exists(SPEAKER_FOLDER):
    os.makedirs(SPEAKER_FOLDER)

LICENSE_FOLDER = os.path.join(STATIC_FOLDER, "LICENSE")
if not os.path.exists(LICENSE_FOLDER):
    os.makedirs(LICENSE_FOLDER)

MODELS_FOLDER = os.path.join(STATIC_FOLDER, "MODELS")
if not os.path.exists(MODELS_FOLDER):
    os.makedirs(MODELS_FOLDER)

if not os.path.exists(f"{MODELS_FOLDER}/text_fail.pkl"):
    hf_hub_download(repo_id="linl03/data_file_config",
                    filename="text_fail.pkl", repo_type="dataset", local_dir=MODELS_FOLDER)
with open(f"{MODELS_FOLDER}/text_fail.pkl", "rb") as file:
    text_fail = pickle.load(file)

file_path = f"{MODELS_FOLDER}/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
if not os.path.isfile(file_path):
    hf_hub_download(repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                    filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", local_dir=MODELS_FOLDER,)


languages = {
    "vi": "Tiếng Việt",
    "cn": "Tiếng Trung",
    "en": "Tiếng Anh",
    "fr": "Tiếng Pháp",
    "de": "Tiếng Đức",
    "ja": "Tiếng Nhật",
    "ko": "Tiếng Hàn",
    "es": "Tiếng Tây Ban Nha",
    "ru": "Tiếng Nga",
    "th": "Tiếng Thái",
    "it": "Tiếng Ý",
    "pt": "Tiếng Bồ Đào Nha",
    "ar": "Tiếng Ả Rập",
    "hi": "Tiếng Hindi",
    "bn": "Tiếng Bengali",
    "tr": "Tiếng Thổ Nhĩ Kỳ",
    "pl": "Tiếng Ba Lan",
    "nl": "Tiếng Hà Lan",
    "sv": "Tiếng Thụy Điển",
    "da": "Tiếng Đan Mạch",
    "fi": "Tiếng Phần Lan",
    "no": "Tiếng Na Uy",
    "el": "Tiếng Hy Lạp",
    "hu": "Tiếng Hungari",
    "cs": "Tiếng Séc",
    "ro": "Tiếng Rumani",
    "sk": "Tiếng Slovak",
    "he": "Tiếng Hebrew",
    "ms": "Tiếng Mã Lai",
    "tl": "Tiếng Tagalog",
    "sw": "Tiếng Swahili",
    "vi": "Tiếng Việt",
    "ja": "Tiếng Nhật",
    "pa": "Tiếng Punjabi",
    "ta": "Tiếng Tamil",
    "te": "Tiếng Telugu",
    "ml": "Tiếng Malayalam",
    "kn": "Tiếng Kannada",
    "lv": "Tiếng Latvia",
    "lt": "Tiếng Litva",
    "et": "Tiếng Estonia",
    "is": "Tiếng Iceland",
    "sq": "Tiếng Albania",
    "bs": "Tiếng Bosnia",
    "sr": "Tiếng Serbia",
    "mk": "Tiếng Macedonia",
    "tl": "Tiếng Tagalog",
    "ky": "Tiếng Kyrgyz",
    "uz": "Tiếng Uzbek",
    "tg": "Tiếng Tajik",
    "am": "Tiếng Amharic",
    "sn": "Tiếng Shona",
    "zu": "Tiếng Zulu",
    "xh": "Tiếng Xhosa",
    "km": "Tiếng Khmer",
    "my": "Tiếng Myanmar",
    "ne": "Tiếng Nepali",
    "si": "Tiếng Sinhala",
    "dv": "Tiếng Dhivehi",
    "ha": "Tiếng Hausa",
    "yi": "Tiếng Yiddish",
    "sm": "Tiếng Samoan",
    "fj": "Tiếng Fiji",
    "na": "Tiếng Nauru",
    "mi": "Tiếng Maori",
    "gw": "Tiếng Galician",
    "ca": "Tiếng Catalan",
    "tl": "Tiếng Tagalog",
    "ku": "Tiếng Kurdish",
    "eo": "Tiếng Esperanto",
    "oc": "Tiếng Occitan",
    "sc": "Tiếng Sardinia",
    "gd": "Tiếng Scots Gaelic",
    "cy": "Tiếng Welsh",
    "gv": "Tiếng Manx",
    "la": "Tiếng Latinh",
    "": "None"
}

NEON_GREEN = '\033[32m'
NEON_CYAN = '\x1b[36m'
RESET_COLOR = '\033[0m'
NEON_RED = '\x1b[31m'
