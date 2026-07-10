"""
LinguaScribe Bot
-----------------
A Telegram bot that transcribes, translates, subtitles, and summarizes
Persian-language audio/video content using the Gemini API.

Designed to run inside Google Colab (see README.md for setup), but the
core logic will run anywhere Python 3.10+ and FFmpeg are available -
just replace the `google.colab.userdata` secret loading in `main()`
with environment variables or a `.env` file.
"""

import os
import asyncio
import nest_asyncio
import datetime
from pathlib import Path
import logging
import re
import math
import zipfile
import shutil  # For robust directory cleanup and copying
from io import BytesIO

# Import for Google Drive
from google.colab import drive

from telethon import TelegramClient, events, Button
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold, GenerationConfig
# For specific error handling
from google.api_core import exceptions as google_exceptions
from pydub import AudioSegment

# --- Configuration ---
# Apply nest_asyncio early
nest_asyncio.apply()

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Google Drive Configuration ---
GDRIVE_ROOT_PATH_STR = "/content/drive"
GDRIVE_MYDRIVE_PATH_STR = f"{GDRIVE_ROOT_PATH_STR}/MyDrive"
GDRIVE_STTCOLAB_FOLDER_NAME = "STTCOLAB"
GDRIVE_STTCOLAB_PATH_STR = f"{GDRIVE_MYDRIVE_PATH_STR}/{GDRIVE_STTCOLAB_FOLDER_NAME}"
GDRIVE_STTCOLAB_PATH = Path(GDRIVE_STTCOLAB_PATH_STR)
GDRIVE_SAVE_ENABLED = False  # Will be set to True if Drive mounts successfully

try:
    from google.colab import userdata
    API_ID = int(userdata.get('TELEGRAM_API_ID'))
    API_HASH = userdata.get('TELEGRAM_API_HASH')
    BOT_TOKEN = userdata.get('TELEGRAM_BOT_TTS')

    GOOGLE_API_KEYS_LIST = []
    for i in range(1, 6):
        key = userdata.get(f'GOOGLE_API_KEY_{i}')
        if key:
            GOOGLE_API_KEYS_LIST.append(key)

    if not all([API_ID, API_HASH, BOT_TOKEN]):
        raise ValueError("Telegram API_ID, API_HASH, or BOT_TOKEN is missing.")
    if not GOOGLE_API_KEYS_LIST:
        raise ValueError(
            "At least one GOOGLE_API_KEY_n must be configured in Colab secrets.")

except Exception as e:
    logger.critical(f"Error loading secrets: {e}")
    exit()

# --- Model and API Configuration ---
MODEL_PREFERENCES = {
    "transcription": [  "gemini-3.1-flash-lite-preview","gemini-3-flash-preview",
        "gemma-4-31b-it",
        "gemma-4-26b-a4b-it",
        "gemini-2.5-flash","gemma-3-27b"],
    "summarization_detailed": ["gemini-3-flash-preview", "gemini-3.1-flash-lite-preview"],
    "translation_segmentation": ["gemma-4-26b-a4b-it","gemma-4-31b-it"],
    "bot_response": ["gemma-4-31b-it","gemma-4-26b-a4b-it","gemini-3-flash-preview"],
}

GENERATION_CONFIGS = {
    "default": GenerationConfig(temperature=0.5),
    "summarization_detailed": GenerationConfig(temperature=0.4, top_p=0.95),
    "translation_segmentation": GenerationConfig(temperature=0.2),
    "bot_response": GenerationConfig(temperature=0.7),
}

SAFETY_SETTINGS = [
    {"category": HarmCategory.HARM_CATEGORY_HARASSMENT,
        "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH},
    {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH},
    {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH},
    {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH},
]

_current_key_index = 0
_active_google_api_key = None


def configure_gemini_client(api_key_to_set):
    global _active_google_api_key
    if api_key_to_set != _active_google_api_key:
        logger.info(
            f"Configuring Google AI SDK with API key ending with ...{api_key_to_set[-4:]}")
        try:
            genai.configure(api_key=api_key_to_set)
            _active_google_api_key = api_key_to_set
            return True
        except Exception as e:
            logger.error(
                f"Failed to configure Google AI SDK with key ...{api_key_to_set[-4:]}: {e}")
            _active_google_api_key = None
            return False
    return True


if not configure_gemini_client(GOOGLE_API_KEYS_LIST[0]):
    logger.critical(
        "Failed to configure Gemini with the initial API key. Exiting.")
    exit()

session_name = f"bot_session_{BOT_TOKEN.split(':')[0]}"
client = TelegramClient(session_name, API_ID, API_HASH)
TEMP_DIR = Path("./temp_files_linguascribe_bot")
TEMP_DIR.mkdir(parents=True, exist_ok=True)
TEMP_EXTRACTION_DIR = TEMP_DIR / "extracted_files"
TEMP_EXTRACTION_DIR.mkdir(parents=True, exist_ok=True)

MAX_DURATION_MINUTES = 25
MAX_DURATION_MS = MAX_DURATION_MINUTES * 60 * 1000
VIDEO_MIME_TYPES = ['video/mp4', 'video/mpeg', 'video/quicktime',
                    'video/x-msvideo', 'video/x-flv', 'video/webm', 'video/x-matroska', 'video/avi']
AUDIO_OUTPUT_FORMAT = "ogg"
AUDIO_OUTPUT_CODEC = "libopus"
AUDIO_OUTPUT_BITRATE = "48k"
AUDIO_SAMPLE_RATE = 16000

# --- Google Drive Helper ---


async def save_to_google_drive(local_file_path_str: str, gdrive_target_dir: Path):
    """Copies a local file to the specified Google Drive directory."""
    global GDRIVE_SAVE_ENABLED
    if not GDRIVE_SAVE_ENABLED:
        logger.info(
            f"Google Drive saving is disabled (likely mount failed). Skipping save for {local_file_path_str}.")
        return

    try:
        local_file = Path(local_file_path_str)
        if not local_file.exists():
            logger.error(
                f"Local file {local_file_path_str} does not exist. Cannot copy to Drive.")
            return

        # Ensure target GDrive directory exists (it should have been created at startup)
        if not gdrive_target_dir.exists():
            logger.warning(
                f"Google Drive target directory {gdrive_target_dir} not found. Attempting to create it.")
            gdrive_target_dir.mkdir(parents=True, exist_ok=True)

        destination_path = gdrive_target_dir / local_file.name

        logger.info(
            f"Attempting to copy {local_file.name} to Google Drive: {destination_path}")
        await asyncio.to_thread(shutil.copy, str(local_file), str(destination_path))
        logger.info(
            f"Successfully copied {local_file.name} to Google Drive: {destination_path}")
    except Exception as e:
        logger.error(
            f"Failed to save {local_file_path_str} to Google Drive directory {gdrive_target_dir}: {e}", exc_info=True)

# --- Helper Functions ---


async def cleanup_files_and_dirs(*paths):
    for path_obj in paths:
        path = Path(path_obj)
        if not path.exists():
            continue
        try:
            if path.is_file():
                path.unlink()
                logger.info(f"Deleted temporary file: {path}")
            elif path.is_dir():
                shutil.rmtree(path)
                logger.info(f"Deleted temporary directory: {path}")
        except OSError as e:
            logger.error(f"Error deleting {path}: {e}")


def generate_srt_with_timecodes(segmented_text):
    lines = [line for line in segmented_text.split("\n") if line.strip()]
    if not lines:
        return "1\n00:00:00,000 --> 00:00:05,000\n(محتوایی برای زمان‌بندی وجود ندارد)\n"
    srt_content = []
    current_time_total_seconds = 0
    segment_duration_seconds = 5

    def format_time(s):
        return f"{int(s // 3600):02}:{int(s % 3600 // 60):02}:{int(s % 60):02},{int((s % 1) * 1000):03}"
    for i, line in enumerate(lines):
        start_seconds = current_time_total_seconds
        end_seconds = current_time_total_seconds + segment_duration_seconds
        srt_content.append(str(i + 1))
        srt_content.append(
            f"{format_time(start_seconds)} --> {format_time(end_seconds)}")
        srt_content.append(line)
        srt_content.append("")
        current_time_total_seconds = end_seconds + 0.5
    return "\n".join(srt_content)


async def get_audio_duration(file_path):
    try:
        audio = AudioSegment.from_file(file_path)
        duration_ms = len(audio)
        logger.info(
            f"Audio duration for {file_path}: {duration_ms/1000:.2f}s ({duration_ms/60000:.2f}min)")
        return duration_ms
    except Exception as e:
        logger.error(
            f"Error getting audio duration for {file_path}: {e}", exc_info=True)
        raise


async def split_audio_file(file_path, base_name, max_duration_ms=MAX_DURATION_MS):
    try:
        audio = AudioSegment.from_file(file_path)
        total_duration_ms = len(audio)

        if total_duration_ms <= max_duration_ms:
            logger.info(f"Audio {base_name} is short enough, no split needed.")
            return [str(file_path)]

        num_chunks = math.ceil(total_duration_ms / max_duration_ms)
        logger.info(f"Splitting audio {base_name} into {num_chunks} chunks.")
        chunk_paths = []
        for i in range(num_chunks):
            start_ms = i * max_duration_ms
            end_ms = min((i + 1) * max_duration_ms, total_duration_ms)
            chunk = audio[start_ms:end_ms]
            chunk_filename = f"{base_name}_part{i+1}.{AUDIO_OUTPUT_FORMAT}"
            chunk_path_obj = TEMP_DIR / chunk_filename
            chunk_path = str(chunk_path_obj)
            logger.info(f"Exporting chunk {i+1}/{num_chunks} to {chunk_path}")
            chunk.export(chunk_path, format=AUDIO_OUTPUT_FORMAT,
                         codec=AUDIO_OUTPUT_CODEC if AUDIO_OUTPUT_FORMAT == "ogg" else None, bitrate=AUDIO_OUTPUT_BITRATE)

            # Verify chunk
            if not chunk_path_obj.exists() or chunk_path_obj.stat().st_size < 100:  # Basic check
                logger.warning(
                    f"Split audio chunk {chunk_path} might be empty or invalid.")

            chunk_paths.append(chunk_path)
        return chunk_paths
    except Exception as e:
        logger.error(
            f"Error splitting audio file {base_name}: {e}", exc_info=True)
        raise


async def gemini_request_with_retry(
    task_name: str,
    model_preference_key: str,
    prompt_parts: list,
    generation_config_key: str,
    file_path_to_upload: str = None
):
    global _current_key_index
    max_key_cycles = len(GOOGLE_API_KEYS_LIST)

    for key_cycle in range(max_key_cycles):
        current_api_key = GOOGLE_API_KEYS_LIST[_current_key_index]
        logger.info(
            f"[Task: {task_name}] Attempting with API key ending ...{current_api_key[-4:]} (Cycle {key_cycle+1}/{max_key_cycles})")

        if not configure_gemini_client(current_api_key):
            _current_key_index = (_current_key_index + 1) % len(GOOGLE_API_KEYS_LIST)
            continue

        models_to_try = MODEL_PREFERENCES.get(
            model_preference_key, [MODEL_PREFERENCES["bot_response"][0]])
        gen_config = GENERATION_CONFIGS.get(
            generation_config_key, GENERATION_CONFIGS["default"])

        # Build final prompt parts with file content if needed
        final_prompt_parts = list(prompt_parts)

        if file_path_to_upload:
            file_to_upload_obj = Path(file_path_to_upload)
            if not file_to_upload_obj.exists():
                logger.error(f"[Task: {task_name}] File does not exist: {file_path_to_upload}")
                raise ValueError(f"File does not exist: {file_path_to_upload}")

            file_size = file_to_upload_obj.stat().st_size
            if file_size < 100:
                logger.warning(f"[Task: {task_name}] File is very small ({file_size} bytes): {file_path_to_upload}")
                # Don't raise, just warn - might be valid for testing

            try:
                logger.info(f"[Task: {task_name}] Reading file: {file_path_to_upload} (size: {file_size} bytes)")

                # Determine MIME type
                file_ext = file_to_upload_obj.suffix.lower()
                mime_type_map = {
                    '.ogg': 'audio/ogg',
                    '.opus': 'audio/ogg',
                    '.mp3': 'audio/mpeg',
                    '.wav': 'audio/wav',
                    '.m4a': 'audio/mp4',
                    '.aac': 'audio/aac',
                    '.flac': 'audio/flac',
                    '.webm': 'audio/webm',
                    '.mp4': 'video/mp4',
                    '.mov': 'video/quicktime',
                    '.avi': 'video/x-msvideo',
                    '.mkv': 'video/x-matroska'
                }
                mime_type = mime_type_map.get(file_ext, 'audio/ogg')
                logger.info(f"[Task: {task_name}] Detected MIME type: {mime_type} for extension: {file_ext}")

                # Read file as bytes (in thread to avoid blocking)
                file_data = await asyncio.to_thread(file_to_upload_obj.read_bytes)
                logger.info(f"[Task: {task_name}] Successfully read {len(file_data)} bytes from file")

                # Import base64 for encoding
                import base64

                # Create inline data part
                file_part = {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": base64.b64encode(file_data).decode('utf-8')
                    }
                }

                # Add file to the beginning of prompt parts
                final_prompt_parts = [file_part] + final_prompt_parts
                logger.info(f"[Task: {task_name}] File content prepared as inline data (base64 encoded)")

            except Exception as e:
                logger.error(f"[Task: {task_name}] Failed to read/encode file: {e}", exc_info=True)
                # Don't skip to next key - this is a file reading error, not an API error
                raise

        for model_name in models_to_try:
            logger.info(
                f"[Task: {task_name}] Trying model: {model_name} with key ...{current_api_key[-4:]}")
            try:
                model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=gen_config,
                    safety_settings=SAFETY_SETTINGS
                )

                logger.info(f"[Task: {task_name}] Sending request to model {model_name}...")
                response = await asyncio.to_thread(
                    model.generate_content,
                    contents=final_prompt_parts
                )
                logger.info(f"[Task: {task_name}] Received response from model {model_name}")

                # Check for blocked content
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    block_reason = response.prompt_feedback.block_reason_message or str(response.prompt_feedback.block_reason)
                    logger.warning(
                        f"[Task: {task_name}] Request blocked for model {model_name}. Reason: {block_reason}")
                    # Try next model
                    continue

                # Check if we have candidates
                if not response.candidates:
                    logger.warning(f"[Task: {task_name}] No candidates in response from model {model_name}")
                    continue

                # Check if we have text
                if not response.text or not response.text.strip():
                    logger.warning(f"[Task: {task_name}] Empty text response from model {model_name}")
                    continue

                logger.info(
                    f"[Task: {task_name}] Successful response from model {model_name} with key ...{current_api_key[-4:]} (length: {len(response.text)} chars)")

                return response.text.strip()

            except (google_exceptions.ResourceExhausted,
                    google_exceptions.InternalServerError,
                    google_exceptions.DeadlineExceeded,
                    google_exceptions.ServiceUnavailable,
                    google_exceptions.Aborted) as e:
                logger.warning(
                    f"[Task: {task_name}] Retryable API error with model {model_name}: {e}.")
                await asyncio.sleep(2)
                continue
            except (google_exceptions.InvalidArgument, google_exceptions.PermissionDenied) as e:
                logger.error(
                    f"[Task: {task_name}] Non-retryable API error with model {model_name}: {e}. Trying next model.")
                continue  # Try next model instead of breaking
            except genai.types.BlockedPromptException as e:
                logger.error(
                    f"[Task: {task_name}] BlockedPromptException with model {model_name}: {e}.")
                raise
            except AttributeError as e:
                if "'GenerateContentResponse' object has no attribute 'text'" in str(e):
                    logger.error(f"[Task: {task_name}] Response has no text attribute. Response object: {response}")
                    # Try to extract text from parts
                    try:
                        if hasattr(response, 'parts'):
                            text = ''.join(part.text for part in response.parts if hasattr(part, 'text'))
                            if text:
                                logger.info(f"[Task: {task_name}] Extracted text from parts")
                                return text.strip()
                    except:
                        pass
                logger.error(f"[Task: {task_name}] AttributeError with model {model_name}: {e}", exc_info=True)
                continue
            except Exception as e:
                logger.error(
                    f"[Task: {task_name}] Unexpected error with model {model_name}: {e}", exc_info=True)
                continue  # Try next model instead of breaking

        _current_key_index = (_current_key_index + 1) % len(GOOGLE_API_KEYS_LIST)
        logger.info(f"[Task: {task_name}] Cycled to next API key")

    logger.error(
        f"[Task: {task_name}] All API keys and models failed after {max_key_cycles} cycles.")
    raise Exception(
        f"Failed to get response for {task_name} after multiple retries.")


async def transcribe_audio_google(file_path):
    logger.info(
        f"Transcribing audio file with speaker diarization attempt: {file_path}")

    # --- IMPROVED PROMPT ---
    # This new prompt is more robust. It asks the model to identify the language first
    # and handles both single and multiple speakers gracefully.
    prompt = """
Transcribe the audio file provided. Follow these instructions carefully:

1. Listen to the ENTIRE audio file
2. Identify the language being spoken
3. Transcribe ALL speech accurately in the original language
4. If multiple speakers are clearly distinguishable, label them as "Speaker 1:", "Speaker 2:", etc.
5. If only one speaker or speakers cannot be distinguished, provide continuous transcription without labels
6. Do NOT translate - transcribe in the original language
7. Do NOT summarize - transcribe everything spoken

Return ONLY the transcription with no additional commentary.
"""

    # Add a sanity check to log the file size before sending
    try:
        file_size = os.path.getsize(file_path)
        logger.info(f"File size for transcription is: {file_size / 1024:.2f} KB")
        if file_size < 256: # Warn if the file is extremely small
             logger.warning("The audio file for transcription is very small. This might result in a short or empty transcription.")
    except OSError as e:
        logger.error(f"Could not get file size for {file_path}: {e}")


    transcription = await gemini_request_with_retry(
        task_name="AudioTranscriptionWithDiarization",
        model_preference_key="transcription",
        prompt_parts=[prompt],
        generation_config_key="default",
        file_path_to_upload=file_path
    )

    if not transcription or transcription.strip() == "":
        logger.warning(
            f"Transcription for {file_path} resulted in empty text.")
        raise ValueError("Transcription failed: The model returned no text.")

    logger.info(f"Transcription (raw output):\n{transcription[:500]}...")
    return transcription

DETAILED_SUMMARY_PROMPT_FOR_COLLEGE_PREP_FA = """
شما یک دستیار آموزشی خبره هستید که وظیفه تهیه مطالب مطالعه برای آمادگی آزمون‌های دانشگاه را بر عهده دارید.
فایل صوتی (که محتوای آن در متن پیاده‌سازی شده آمده) و متن پیاده‌سازی شده اولیه آن ارائه شده است.
لطفاً این محتوا را با دقت بسیار بالا تحلیل کرده و یک خلاصه بسیار جامع، دقیق و با جزئیات فراوان به زبان فارسی روان تهیه کنید که برای دانشجویی که نیاز به یادآوری و درک کامل این اطلاعات برای یک آزمون مهم دارد، مناسب باشد.

متن پیاده‌سازی شده اولیه (برای کمک به زمینه و کلمات کلیدی):
\"\"\"
{transcription_context}
\"\"\"

دستورالعمل‌های خلاصه‌سازی جامع برای آمادگی آزمون:

1.  **مقدمه و هدف کلی (۱-۲ پاراگراف):**
    *   موضوع اصلی و هدف کلی از ارائه این محتوا چیست؟
    *   زمینه و بستر اصلی بحث چیست؟

2.  **مفاهیم و تعاریف کلیدی (لیست شماره‌گذاری شده):**
    *   تمامی اصطلاحات، مفاهیم و واژگان تخصصی مهم مطرح شده را شناسایی کنید.
    *   هر کدام را به طور واضح و دقیق در چارچوب بحث تعریف کنید.

3.  **نکات اصلی و استدلال‌ها (ساختار درختی یا با عنوان‌بندی مناسب با استفاده از Markdown):**
    *   تمامی نکات، ایده‌ها و استدلال‌های اصلی مطرح شده را به تفصیل بیان کنید.
    *   برای هر نکته یا استدلال، شواهد، مثال‌ها، آمار، ارقام، تاریخ‌ها و اسامی مهم ذکر شده را به طور کامل بیاورید.
    *   اگر زنجیره منطقی یا مراحل خاصی در استدلال‌ها وجود دارد، آن‌ها را گام به گام توضیح دهید.

4.  **جزئیات تکمیلی و مثال‌های مهم (حداقل ۵-۷ مورد یا بیشتر در صورت لزوم):**
    *   مثال‌های کلیدی، موارد خاص، مطالعات موردی یا نمونه‌هایی که برای روشن شدن مفاهیم ارائه شده‌اند را با جزئیات شرح دهید.
    *   نقل قول‌های مهم و تاثیرگذار را (در صورت وجود) با ذکر دقیق آورده و اهمیت آن‌ها را توضیح دهید.

5.  **تحلیل عمیق محتوا (در صورت امکان و مرتبط بودن):**
    *   ارتباط بین مفاهیم مختلف چگونه است؟
    *   نقاط قوت و ضعف استدلال‌های ارائه شده (در صورت تحلیل در خود محتوا) چیست؟
    *   پیشنهادات، راهکارها یا نتایج عملی که از بحث حاصل می‌شود، کدامند؟
    *   هرگونه پیش‌فرض، فرضیه زمینه‌ای یا پیامدهای پنهان را شناسایی کنید.

6.  **نتیجه‌گیری اصلی و پیام نهایی (۱-۲ پاراگراف):**
    *   جمع‌بندی نهایی بحث و مهمترین نتایجی که می‌توان گرفت چیست؟
    *   پیام اصلی یا درسی که مخاطب باید از این محتوا بگیرد چیست؟

7.  **ساختار و زبان:**
    *   خلاصه باید کاملاً به زبان فارسی رسمی، علمی و روان باشد.
    *   از ساختار منطقی با عنوان‌بندی و شماره‌گذاری مناسب (مانند لیست‌ها، تیترهای فرعی با استفاده از Markdown مانند #, ##, ###, *, -) برای سازماندهی اطلاعات استفاده کنید تا خوانایی و قابلیت مرور آن برای مطالعه افزایش یابد.
    *   در ارائه جزئیات کوتاهی نکنید؛ هدف، پوشش کامل و عمیق مطالب برای آمادگی آزمون است.
    *   فقط و فقط خلاصه نهایی مطابق ساختار درخواستی، بدون عبارت مقدماتی یا توضیحات اضافی درباره فرآیند خلاصه‌سازی ارائه شود.
"""


async def summarize_audio_google(transcription_context):
    logger.info("Generating detailed summary for college prep...")
    if not transcription_context or len(transcription_context.strip()) < 10:
        logger.warning(
            f"Transcription context for summary is too short: '{transcription_context}'.")
        return ("\u200F" + "با احترام، به عنوان یک دستیار آموزشی خبره برای آمادگی کنکور، وظیفه من تهیه خلاصه‌ای جامع و دقیق از محتوای ارائه شده است.\n\n"
                f"متن پیاده‌سازی شده‌ای که ارائه کرده‌اید ('{transcription_context}') بسیار کوتاه است یا فاقد اطلاعات، مفاهیم، استدلال‌ها، مثال‌ها، آمار، تاریخ‌ها یا جزئیات کافی است که بتوان بر اساس آن یک خلاصه جامع، دقیق و با جزئیات فراوان مطابق با ساختار درخواستی تهیه کرد.\n\n"
                "برای تولید خلاصه‌ای که برای آمادگی آزمون ورودی دانشگاه مفید باشد، نیاز به محتوای متنی کامل‌تر و پرمحتواتری از فایل صوتی مربوطه است.\n\n"
                "لطفاً از ارسال فایل صوتی با محتوای کافی اطمینان حاصل کنید.")

    summary_prompt_formatted = DETAILED_SUMMARY_PROMPT_FOR_COLLEGE_PREP_FA.format(
        transcription_context=transcription_context)

    # The summary is purely text-based, so no file is needed here.
    summary_text = await gemini_request_with_retry(
        task_name="DetailedSummarization",
        model_preference_key="summarization_detailed",
        prompt_parts=[summary_prompt_formatted],
        generation_config_key="summarization_detailed",
        file_path_to_upload=None  # Explicitly no file
    )

    if not summary_text:
        raise ValueError(
            "Summarization failed: No text returned after retries.")
    return "\u200F" + summary_text


async def translate_to_persian_google(text):
    if not text or not text.strip():
        return ""
    logger.info("Translating text to Persian...")
    prompt = f'Translate the following text to Persian:\n\n"{text}"\n\nReturn ONLY the Persian translation, with no introductory phrases.'

    # CORRECTED: This now expects only one return value
    translation = await gemini_request_with_retry(
        task_name="TranslationToPersian",
        model_preference_key="translation_segmentation",
        prompt_parts=[prompt],
        generation_config_key="translation_segmentation"
    )

    if not translation:
        raise ValueError("Translation failed: No text returned.")
    return translation


async def segment_persian_text_google(persian_text):
    logger.info("Segmenting Persian text for SRT...")
    segmentation_prompt = f"""Take the following Persian text and break it into suitable subtitle segments. Each segment should be on a new line. Aim for natural breaks and readable lengths for subtitles (typically 1-2 short sentences or phrases).
Return ONLY the segmented text, with each segment on a new line. Do not add numbering.
Persian text:
---
{persian_text}
---"""

    # CORRECTED: This now expects only one return value
    segmented_text = await gemini_request_with_retry(
        task_name="TextSegmentation",
        model_preference_key="translation_segmentation",
        prompt_parts=[segmentation_prompt],
        generation_config_key="translation_segmentation"
    )

    if not segmented_text:
        logger.warning(
            "LLM Segmentation response was empty. Using regex fallback.")
        segments = re.split(r'[।\.؟!\n]+', persian_text)
        segmented_text = "\n".join(s.strip() for s in segments if s.strip())
        if not segmented_text:
            raise ValueError(
                "Segmentation failed: No text from LLM or fallback.")
    return segmented_text


async def generate_persian_srt_google(transcription):
    logger.info("Generating Persian SRT...")
    try:
        # Check for minimal content
        if not transcription or len(transcription.strip()) < 5:
            logger.warning(
                f"Transcription for SRT generation is too short: '{transcription}'. Skipping SRT.")
            return None  # Or an empty SRT
        persian_translation = await translate_to_persian_google(transcription)
        if not persian_translation:
            raise ValueError("Translation step failed for SRT.")
        segmented_persian_text = await segment_persian_text_google(persian_translation)
        if not segmented_persian_text:
            raise ValueError("Segmentation step failed for SRT.")
        srt_content = generate_srt_with_timecodes(segmented_persian_text)
        logger.info("SRT generation successful.")
        return srt_content
    except Exception as e:
        logger.error(f"Error generating SRT: {e}", exc_info=True)
        raise


async def get_bot_response_google(message_text):
    logger.info(f"Getting bot response for: {message_text[:50]}...")
    prompt = f"""You are LinguaScribe_Bot, a helpful Telegram assistant. The user's language is Persian.
User says: "{message_text}"
Provide a concise and helpful response in Persian.
If they ask about your capabilities, mention:
- پیاده‌سازی صوت به متن (Audio transcription)
- خلاصه‌سازی جامع و تخصصی محتوای صوتی (Detailed audio summarization for study/prep)
- تولید فایل زیرنویس SRT به فارسی (Persian SRT subtitle generation)
- پردازش فایل‌های ZIP حاوی صوت یا متن (Processing ZIP files with audio/text)
- تبدیل ویدیو به صوت برای پردازش (Video to audio conversion for processing)
- ذخیره نتایج متنی (خلاصه و پیاده‌سازی) در گوگل درایو (Saves text results to Google Drive)

Keep responses brief. If the input is non-sensical or just a greeting, respond politely and briefly in Persian.
Return ONLY the bot's reply in Persian.
"""

    # CORRECTED: This now expects only one return value
    reply = await gemini_request_with_retry(
        task_name="BotResponse",
        model_preference_key="bot_response",
        prompt_parts=[prompt],
        generation_config_key="bot_response"
    )

    if not reply:
        return "متاسفانه در حال حاضر قادر به پاسخگویی نیستم."
    return reply


async def process_single_audio_file_operations(
    audio_file_path: str,
    original_name_base: str,
    chat_id: int,
    processing_msg_event,
    is_part_of_long_audio=False
):
    files_to_cleanup_later = []
    original_transcription = ""
    transcription_path_str = None
    srt_path_str = None
    summary_path_str = None

    try:
        # Step 1: Transcribe in the original language for maximum accuracy
        await client.edit_message(processing_msg_event, processing_msg_event.text + "\n\n⏳ در حال پیاده‌سازی متن (زبان اصلی)...")
        original_transcription = await transcribe_audio_google(audio_file_path)
        files_to_cleanup_later.append(audio_file_path)

        if not original_transcription:
            raise ValueError("The initial transcription was empty.")

        # Step 2: Translate the accurate transcription into Persian
        await client.edit_message(processing_msg_event, processing_msg_event.text + "\n⏳ در حال ترجمه متن به فارسی...")
        persian_transcription = await translate_to_persian_google(original_transcription)

        if not persian_transcription:
            raise ValueError("Translation to Persian failed or returned empty.")

        # Step 3: Save and send the PERSIAN transcription
        transcription_filename = f"{original_name_base}_transcription_fa.txt"
        transcription_path = TEMP_DIR / transcription_filename
        with open(transcription_path, "w", encoding="utf-8") as f:
            f.write(persian_transcription)
        transcription_path_str = str(transcription_path)
        files_to_cleanup_later.append(transcription_path_str)
        await save_to_google_drive(transcription_path_str, GDRIVE_STTCOLAB_PATH)

        await client.send_file(chat_id, transcription_path_str, caption="🎤 متن پیاده‌سازی شده (فارسی):")
        await client.edit_message(processing_msg_event, processing_msg_event.text + "\n✅ متن فارسی ارسال شد.")

        # For single files, generate SRT and Summary immediately
        if not is_part_of_long_audio:
            # Step 4: Generate SRT from the ORIGINAL transcription (the SRT function does its own translation)
            await client.edit_message(processing_msg_event, processing_msg_event.text + "\n\n⏳ در حال تولید زیرنویس (SRT)...")
            srt_content = await generate_persian_srt_google(original_transcription)
            if srt_content:
                srt_filename = f"{original_name_base}_subtitles.srt"
                srt_path = TEMP_DIR / srt_filename
                with open(srt_path, "w", encoding="utf-8") as f: f.write(srt_content)
                srt_path_str = str(srt_path)
                files_to_cleanup_later.append(srt_path_str)
                await client.send_file(chat_id, srt_path_str, caption="🎬 فایل زیرنویس (SRT):")
                await client.edit_message(processing_msg_event, processing_msg_event.text + "\n✅ فایل زیرنویس (SRT) ارسال شد.")
            else:
                await client.edit_message(processing_msg_event, processing_msg_event.text + "\n⚠️ محتوای کافی برای تولید زیرنویس (SRT) وجود نداشت.")

            # Step 5: Generate Summary from the PERSIAN transcription
            await client.edit_message(processing_msg_event, processing_msg_event.text + "\n\n⏳ در حال تهیه خلاصه جامع...")
            summary = await summarize_audio_google(persian_transcription)
            summary_filename = f"{original_name_base}_summary.md"
            summary_path = TEMP_DIR / summary_filename
            with open(summary_path, "w", encoding="utf-8") as f: f.write(summary)
            summary_path_str = str(summary_path)
            files_to_cleanup_later.append(summary_path_str)
            await save_to_google_drive(summary_path_str, GDRIVE_STTCOLAB_PATH)
            await client.send_file(chat_id, summary_path_str, caption="📝 *خلاصه جامع محتوا:*", parse_mode='md')
            await client.edit_message(processing_msg_event, processing_msg_event.text + "\n✅ خلاصه جامع ارسال شد.")
            await client.edit_message(processing_msg_event, "✅ پردازش فایل صوتی با موفقیت تکمیل شد!")

        # IMPORTANT: Return the ORIGINAL transcription for the parent function to combine if needed
        return original_transcription, transcription_path_str, srt_path_str, summary_path_str, files_to_cleanup_later

    except (genai.types.BlockedPromptException, ValueError, Exception) as e:
        error_message = f"Error in processing: {str(e)}"
        logger.error(f"{error_message} for {original_name_base}", exc_info=True)
        try:
            await client.edit_message(processing_msg_event, f"❌ خطا در پردازش فایل صوتی ({original_name_base}): {str(e)[:200]}")
        except:
            pass # Avoid errors if message can't be edited
        return None, None, None, None, files_to_cleanup_later


async def process_audio_file(event, audio_file_path: str, original_name_base: str, chat_id: int, processing_msg_event):
    all_local_files_to_cleanup = [audio_file_path]

    try:
        audio_duration_ms = await get_audio_duration(audio_file_path)
        needs_splitting = audio_duration_ms > MAX_DURATION_MS
        chunk_paths = []

        if needs_splitting:
            await client.edit_message(
                processing_msg_event,
                f"⚠️ فایل صوتی شما طولانی است ({audio_duration_ms/60000:.1f} دقیقه). در حال تقسیم به قطعات ~{MAX_DURATION_MINUTES} دقیقه‌ای و پردازش..."
            )
            chunk_paths = await split_audio_file(audio_file_path, original_name_base, MAX_DURATION_MS)
            all_local_files_to_cleanup.extend(chunk_paths)
        else:
            chunk_paths = [audio_file_path]

        full_transcription_parts = []

        for i, chunk_path in enumerate(chunk_paths):
            chunk_name_base = f"{original_name_base}_part{i+1}" if needs_splitting else original_name_base
            status_update_msg = f"\n\n⏳ پردازش قطعه {i+1}/{len(chunk_paths)}: {Path(chunk_path).name}"
            current_text = (await client.get_messages(chat_id, ids=processing_msg_event.id)).text
            if len(current_text) > 3800:
                current_text = "⏳ به روز رسانی پردازش..."
            await client.edit_message(processing_msg_event, current_text + status_update_msg)

            # process_single_audio_file_operations returns the ORIGINAL language transcription
            transcription, _, _, _, chunk_cleanup_files = await process_single_audio_file_operations(
                audio_file_path=chunk_path,
                original_name_base=chunk_name_base,
                chat_id=chat_id,
                processing_msg_event=processing_msg_event,
                is_part_of_long_audio=needs_splitting
            )
            all_local_files_to_cleanup.extend(chunk_cleanup_files)

            if transcription:
                full_transcription_parts.append(transcription)
                logger.info(f"✓ Chunk {i+1}/{len(chunk_paths)} transcription collected (length: {len(transcription)} chars)")
            else:
                logger.warning(f"✗ Chunk {i+1} ({Path(chunk_path).name}) failed. Skipping.")
                await client.edit_message(processing_msg_event, processing_msg_event.text + f"\n⚠️ خطایی در پردازش قطعه {i+1} رخ داد.")

        if not full_transcription_parts:
            await client.edit_message(processing_msg_event, "❌ پردازش هیچ بخشی از فایل صوتی موفقیت آمیز نبود.")
            return

        # If the file was split, we now combine, translate, and process the full versions
        if needs_splitting:
            # Combine all transcription parts
            full_original_transcription = "\n\n".join(full_transcription_parts)
            logger.info(f"✓ Combined full transcription: {len(full_original_transcription)} chars from {len(full_transcription_parts)} parts")

            # Translate the full combined text to Persian
            await client.edit_message(processing_msg_event, processing_msg_event.text + "\n\n⏳ در حال ترجمه متن کامل به فارسی...")
            try:
                full_persian_transcription = await translate_to_persian_google(full_original_transcription)
                logger.info(f"✓ Full Persian translation completed: {len(full_persian_transcription)} chars")
            except Exception as e:
                logger.error(f"✗ Failed to translate full text: {e}")
                await client.edit_message(processing_msg_event, processing_msg_event.text + f"\n❌ خطا در ترجمه متن کامل: {str(e)[:100]}")
                return

            # Save and send the full PERSIAN transcription
            full_transcription_filename = f"{original_name_base}_FULL_transcription_fa.txt"
            full_transcription_path = TEMP_DIR / full_transcription_filename

            try:
                with open(full_transcription_path, "w", encoding="utf-8") as f:
                    f.write(full_persian_transcription)
                logger.info(f"✓ Saved full Persian transcription to: {full_transcription_path}")

                all_local_files_to_cleanup.append(str(full_transcription_path))
                await save_to_google_drive(str(full_transcription_path), GDRIVE_STTCOLAB_PATH)
                await client.send_file(chat_id, str(full_transcription_path), caption="🎤 متن کامل پیاده‌سازی شده (فارسی):")
                await client.edit_message(processing_msg_event, processing_msg_event.text + "\n✅ متن کامل فارسی ارسال شد.")
            except Exception as e:
                logger.error(f"✗ Failed to save/send full transcription: {e}")
                await client.edit_message(processing_msg_event, processing_msg_event.text + f"\n⚠️ خطا در ذخیره/ارسال متن کامل")

            # Generate combined SRT from the ORIGINAL full text
            await client.edit_message(processing_msg_event, processing_msg_event.text + "\n\n⏳ در حال تولید زیرنویس (SRT) کامل...")
            try:
                combined_srt_content = await generate_persian_srt_google(full_original_transcription)
                if combined_srt_content:
                    combined_srt_filename = f"{original_name_base}_FULL_subtitles.srt"
                    combined_srt_path = TEMP_DIR / combined_srt_filename
                    with open(combined_srt_path, "w", encoding="utf-8") as f:
                        f.write(combined_srt_content)
                    all_local_files_to_cleanup.append(str(combined_srt_path))
                    await client.send_file(chat_id, str(combined_srt_path), caption="🎬 فایل زیرنویس کامل (SRT):")
                    await client.edit_message(processing_msg_event, processing_msg_event.text + "\n✅ زیرنویس کامل ارسال شد.")
                else:
                    logger.warning("Combined SRT generation returned empty content")
                    await client.edit_message(processing_msg_event, processing_msg_event.text + "\n⚠️ محتوای کافی برای تولید زیرنویس کامل وجود نداشت.")
            except Exception as e:
                logger.error(f"✗ Failed to generate combined SRT: {e}")
                await client.edit_message(processing_msg_event, processing_msg_event.text + f"\n⚠️ خطا در تولید زیرنویس کامل")

            # Generate combined Summary from the PERSIAN full text
            await client.edit_message(processing_msg_event, processing_msg_event.text + "\n\n⏳ در حال تهیه خلاصه جامع کامل...")
            try:
                combined_summary = await summarize_audio_google(full_persian_transcription)
                logger.info(f"✓ Summary generated: {len(combined_summary)} chars")

                summary_filename = f"{original_name_base}_FULL_summary.md"
                summary_path = TEMP_DIR / summary_filename
                with open(summary_path, "w", encoding="utf-8") as f:
                    f.write(combined_summary)
                all_local_files_to_cleanup.append(str(summary_path))
                await save_to_google_drive(str(summary_path), GDRIVE_STTCOLAB_PATH)
                await client.send_file(chat_id, str(summary_path), caption="📝 *خلاصه جامع محتوای کامل:*", parse_mode='md')
                await client.edit_message(processing_msg_event, processing_msg_event.text + "\n✅ خلاصه جامع کامل ارسال شد.")
            except Exception as e:
                logger.error(f"✗ Failed to generate/send summary: {e}")
                await client.edit_message(processing_msg_event, processing_msg_event.text + f"\n⚠️ خطا در تهیه خلاصه جامع")

        final_message = "✅ پردازش فایل با موفقیت تکمیل شد!"
        await client.edit_message(processing_msg_event, final_message)

    except Exception as e:
        logger.exception(f"Critical error in process_audio_file for {original_name_base}: {e}")
        try:
            await client.edit_message(processing_msg_event, f"\n❌ خطای جدی در پردازش فایل صوتی: {str(e)[:100]}")
        except:
            pass
    finally:
        unique_cleanup_paths = list(set(all_local_files_to_cleanup))
        await cleanup_files_and_dirs(*unique_cleanup_paths)

# --- Main Bot Event Handlers ---


@client.on(events.NewMessage(pattern='/start'))
async def start(event):
    sender = await event.get_sender()
    logger.info(f"✓✓✓ /start command received from User {sender.id} in Chat {event.chat_id}")
    drive_status = "فعال" if GDRIVE_SAVE_ENABLED else "غیرفعال (خطا در اتصال)"
    try:
        await event.reply(
            "👋 سلام! به ربات *LinguaScribe* خوش آمدید.\n\n"
            "این ربات می‌تواند فایل‌های صوتی، ویدیویی یا ZIP را پردازش کند:\n"
            "🎤 **پیاده‌سازی متن دقیق**\n"
            "📝 **خلاصه‌سازی جامع و تخصصی** (مناسب آمادگی آزمون)\n"
            "🎬 **تولید زیرنویس SRT فارسی**\n"
            "📹 **تبدیل ویدیو به صوت** برای تحلیل\n"
            "🗜️ **پردازش فایل‌های ZIP** حاوی صوت یا متن\n"
            f"💾 **ذخیره خودکار نتایج در گوگل درایو شما (پوشه STTCOLAB): {drive_status}**\n\n"
            "یک فایل صوتی (voice, audio), ویدیویی, یا فایل ZIP برای من ارسال کنید.",
            parse_mode='md'
        )
        logger.info(f"✓✓✓ /start response sent successfully")
    except Exception as e:
        logger.error(f"✗✗✗ Error in /start handler: {e}", exc_info=True)


@client.on(events.NewMessage(pattern='/help'))
async def help_command(event):
    drive_status = "فعال" if GDRIVE_SAVE_ENABLED else "غیرفعال (خطا در اتصال)"
    await event.reply(
        "🔍 **راهنمای LinguaScribe Bot**\n\n"
        "1️⃣ یک فایل صوتی، ویدیویی، یا ZIP ارسال کنید.\n"
        "2️⃣ ربات به صورت خودکار آن را پردازش می‌کند.\n"
        "   - ویدیو به صوت تبدیل می‌شود.\n"
        "   - فایل‌های ZIP استخراج و محتوای پشتیبانی شده (صوت/متن) پردازش می‌شود.\n"
        "   - برای صوت: متن پیاده‌سازی، خلاصه جامع، و زیرنویس SRT ارائه می‌شود.\n"
        "   - برای متن (از ZIP): محتوا ترکیب و ارسال می‌شود.\n\n"
        f"💾 **ذخیره‌سازی گوگل درایو:** نتایج متنی (فایل‌های .txt و .md) به طور خودکار در پوشه `STTCOLAB` در My Drive شما ذخیره می‌شوند. وضعیت فعلی: {drive_status}\n\n"
        "📋 **نکات**:\n"
        f"• فایل‌های صوتی تا {MAX_DURATION_MINUTES} دقیقه به صورت یکجا، طولانی‌تر به صورت بخش‌بندی شده پردازش می‌شوند.\n"
        "• زبان اصلی فارسی است.\n\n"
        "📌 **دستورات:** /start, /help",
        parse_mode='md'
    )


@client.on(events.NewMessage(func=lambda e: e.text and not e.text.startswith('/')))
async def handle_text_message(event):
    chat_id = event.chat_id
    message_text = event.text
    logger.info(f"Text message in chat {chat_id}: {message_text[:50]}...")
    processing_msg = await event.reply("⏳ در حال پردازش پیام شما...")
    try:
        bot_response = await get_bot_response_google(message_text)
        await client.edit_message(processing_msg, bot_response)
    except Exception as e:
        logger.error(f"Error handling text message: {e}", exc_info=True)
        await client.edit_message(processing_msg, "❌ متأسفانه در پردازش پیام شما مشکلی پیش آمد.")


@client.on(events.NewMessage(func=lambda e: e.audio or e.voice or e.document))
async def handle_media_message(event):
    chat_id = event.chat_id
    sender = await event.get_sender()
    message_id = event.message.id
    logger.info(
        f"Media from User {sender.id} (msg_id:{message_id}) in Chat {chat_id}")

    media_item = None
    file_name_attr = "unknown_file"  # Default
    mime_type_attr = None
    media_type_for_log = "unknown"

    if event.audio:
        media_item = event.audio
        media_type_for_log = "audio"
        mime_type_attr = getattr(media_item, 'mime_type', 'audio/ogg')
        # Try to get filename from attributes, fallback for older clients/formats
        doc_attributes = getattr(media_item, 'attributes', [])
        file_name_from_attr = ""
        if doc_attributes:
            for attr in doc_attributes:
                if hasattr(attr, 'file_name'):
                    file_name_from_attr = attr.file_name
                    break
        file_name_attr = file_name_from_attr or f"audio_{message_id}.{mime_type_attr.split('/')[-1] or 'ogg'}"
    elif event.voice:
        media_item = event.voice
        media_type_for_log = "voice"
        mime_type_attr = getattr(media_item, 'mime_type', 'audio/ogg')
        file_name_attr = f"voice_{message_id}.ogg"  # Voices are usually ogg
    elif event.document:
        media_item = event.document
        mime_type_attr = getattr(media_item, 'mime_type', '')
        doc_attributes = getattr(media_item, 'attributes', [])
        file_name_from_attr = ""
        if doc_attributes:
            for attr in doc_attributes:
                if hasattr(attr, 'file_name'):
                    file_name_from_attr = attr.file_name
                    break
        # Fallback if no filename attribute
        file_name_attr = file_name_from_attr or f"document_{message_id}"

        if mime_type_attr.startswith('audio/'):
            media_type_for_log = "document_audio"
        # Check extension too
        elif mime_type_attr.startswith('video/') or file_name_attr.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv')):
            media_type_for_log = "document_video"
        elif mime_type_attr in ['application/zip', 'application/x-zip-compressed'] or file_name_attr.lower().endswith('.zip'):
            media_type_for_log = "document_zip"
        else:
            logger.warning(
                f"Unsupported document type: {file_name_attr}, MIME: {mime_type_attr}")
            await event.reply("⚠️ این نوع فایل توسط ربات پشتیبانی نمی‌شود. لطفاً فایل صوتی، ویدیویی یا ZIP ارسال کنید.")
            return
    else:
        return

    processing_msg = await event.reply("⏳ در حال دریافت و بررسی فایل...")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    original_name_base = Path(file_name_attr).stem
    # Sanitize original_name_base to prevent issues with creating filenames
    original_name_base = re.sub(
        r'[^\w\.-]', '_', original_name_base) if original_name_base else "file"

    download_path_initial_obj = TEMP_DIR / \
        f"{original_name_base}_{timestamp}{Path(file_name_attr).suffix or '.dat'}"
    download_path_initial = str(download_path_initial_obj)
    local_files_to_cleanup_main_handler = [download_path_initial]

    try:
        logger.info(
            f"Attempting to download: {file_name_attr} to {download_path_initial}")
        await client.download_media(message=event.message, file=download_path_initial)
        if not download_path_initial_obj.exists() or download_path_initial_obj.stat().st_size == 0:
            logger.error(
                f"Failed to download or downloaded empty file: {download_path_initial}")
            await client.edit_message(processing_msg, "❌ دانلود فایل ناموفق بود یا فایل خالی است.")
            return
        logger.info(
            f"File {file_name_attr} ({media_type_for_log}) downloaded to {download_path_initial}")
        await client.edit_message(processing_msg, "✅ فایل دریافت شد. در حال پردازش اولیه...")

        if media_type_for_log == "document_zip":
            await client.edit_message(processing_msg, processing_msg.text + "\n🗜️ فایل ZIP شناسایی شد، در حال استخراج...")
            extraction_path = TEMP_EXTRACTION_DIR / \
                f"{original_name_base}_{timestamp}_extracted"
            extraction_path.mkdir(parents=True, exist_ok=True)
            local_files_to_cleanup_main_handler.append(str(extraction_path))

            try:
                with zipfile.ZipFile(download_path_initial, 'r') as zip_ref:
                    zip_ref.extractall(extraction_path)
                logger.info(f"ZIP extracted to {extraction_path}")
                combined_text_content = []
                audio_files_in_zip = []

                for item in extraction_path.rglob('*'):
                    if item.is_file():
                        item_suffix_lower = item.suffix.lower()
                        # More comprehensive audio check
                        if item_suffix_lower in ['.mp3', '.wav', '.ogg', '.m4a', '.opus', '.flac', '.aac', '.amr', '.oga', '.webm', '.aac'] or \
                           (hasattr(AudioSegment, 'from_file') and AudioSegment.from_file(str(item), warn_if_empty=False) is not None):  # Tentative pydub check
                            audio_files_in_zip.append(item)
                        # Expanded text types
                        elif item_suffix_lower in ['.txt', '.md', '.rtf', '.json', '.csv', '.xml', '.html']:
                            try:
                                combined_text_content.append(
                                    item.read_text(encoding='utf-8', errors='ignore'))
                                logger.info(f"Read text from {item.name}")
                            except Exception as e:
                                logger.warning(
                                    f"Could not read text file {item.name}: {e}")

                if audio_files_in_zip:
                    await client.edit_message(processing_msg, processing_msg.text + f"\n🔊 {len(audio_files_in_zip)} فایل صوتی در ZIP یافت شد. پردازش آن‌ها...")
                    for i, audio_item_path in enumerate(audio_files_in_zip):
                        zip_audio_name_base = f"{original_name_base}_zip_audio{i+1}_{audio_item_path.stem}"
                        zip_audio_name_base = re.sub(
                            r'[^\w\.-]', '_', zip_audio_name_base)  # Sanitize
                        audio_proc_msg_text = f"⏳ پردازش فایل صوتی {i+1}/{len(audio_files_in_zip)} از ZIP: {audio_item_path.name}"
                        audio_proc_msg = await event.reply(audio_proc_msg_text)
                        try:
                            await process_audio_file(event, str(audio_item_path), zip_audio_name_base, chat_id, audio_proc_msg)
                        except Exception as e_zip_audio:
                            logger.error(
                                f"Error processing audio from ZIP {audio_item_path.name}: {e_zip_audio}")
                            await client.edit_message(audio_proc_msg, audio_proc_msg_text + f"\n❌ خطا: {e_zip_audio}")

                if combined_text_content:
                    full_extracted_text = "\n\n--- (محتوای فایل بعدی) ---\n\n".join(
                        combined_text_content)
                    text_output_filename = f"{original_name_base}_extracted_texts.txt"
                    text_output_path = TEMP_DIR / text_output_filename
                    with open(text_output_path, "w", encoding="utf-8") as f:
                        f.write(full_extracted_text)
                    local_files_to_cleanup_main_handler.append(
                        str(text_output_path))
                    await save_to_google_drive(str(text_output_path), GDRIVE_STTCOLAB_PATH)
                    await client.send_file(chat_id, str(text_output_path), caption="📜 متن‌های استخراج شده از فایل ZIP:")
                    await client.edit_message(processing_msg, processing_msg.text + "\n📜 محتوای متنی از ZIP ارسال شد.")

                if not audio_files_in_zip and not combined_text_content:
                    await client.edit_message(processing_msg, processing_msg.text + "\n⚠️ هیچ فایل صوتی یا متنی قابل پردازشی در ZIP یافت نشد.")
                else:
                    await client.edit_message(processing_msg, processing_msg.text + "\n✅ پردازش محتوای ZIP تکمیل شد.")

            except zipfile.BadZipFile:
                logger.error(f"Bad ZIP file: {download_path_initial}")
                await client.edit_message(processing_msg, "❌ فایل ZIP نامعتبر است.")
            except Exception as e:
                logger.error(f"Error processing ZIP file: {e}", exc_info=True)
                await client.edit_message(processing_msg, f"❌ خطا در پردازش فایل ZIP: {e}")
            return

        actual_audio_file_to_process = download_path_initial
        video_converted_to_audio_successfully = False

        if media_type_for_log.startswith("document_video") or mime_type_attr.startswith("video/"):
            await client.edit_message(processing_msg, processing_msg.text + "\n📹 فایل ویدیویی شناسایی شد، در حال تبدیل به صوت فشرده...")
            converted_audio_path_obj = TEMP_DIR / \
                f"{original_name_base}_{timestamp}_converted.{AUDIO_OUTPUT_FORMAT}"
            converted_audio_path = str(converted_audio_path_obj)
            try:
                logger.info(
                    f"Attempting to convert video: {download_path_initial} to {converted_audio_path}")
                video_segment = AudioSegment.from_file(download_path_initial)

                if len(video_segment) == 0:
                    logger.error(
                        f"Video to audio conversion for '{download_path_initial}' resulted in an empty audio segment. Ensure FFmpeg is available and the video has an audio track.")
                    raise ValueError(
                        "پردازش ویدیو ناموفق بود: ترک صوتی استخراج شده خالی است. لطفاً بررسی کنید ویدیو دارای صوت باشد و FFmpeg در دسترس باشد.")

                audio_only = video_segment.set_channels(
                    1).set_frame_rate(AUDIO_SAMPLE_RATE)
                audio_only.export(
                    converted_audio_path, format=AUDIO_OUTPUT_FORMAT,
                    codec=AUDIO_OUTPUT_CODEC if AUDIO_OUTPUT_FORMAT == "ogg" else None,
                    bitrate=AUDIO_OUTPUT_BITRATE
                )
                if not converted_audio_path_obj.exists() or converted_audio_path_obj.stat().st_size < 256:
                    check_segment = AudioSegment.from_file(
                        converted_audio_path)  # Will raise if invalid
                    if len(check_segment) == 0:
                        logger.error(
                            f"Exported audio file '{converted_audio_path}' from video is effectively empty.")
                        raise ValueError(
                            f"فایل صوتی تبدیل شده از ویدیو ('{converted_audio_path_obj.name}') خالی یا نامعتبر است.")

                actual_audio_file_to_process = converted_audio_path
                local_files_to_cleanup_main_handler.append(
                    converted_audio_path)
                video_converted_to_audio_successfully = True
                logger.info(
                    f"Video successfully converted to audio: {converted_audio_path}, duration: {len(audio_only)/1000.0:.2f}s")
                await client.edit_message(processing_msg, processing_msg.text + "\n✅ ویدیو به صوت تبدیل شد.")
            except Exception as e:
                logger.error(
                    f"Error converting video to audio ('{download_path_initial}'): {e}", exc_info=True)
                user_error_message = f"❌ خطا در تبدیل ویدیو به صوت ({Path(download_path_initial).name}): {str(e)}. "
                if "ffmpeg" in str(e).lower() or "avconv" in str(e).lower() or "could not find ffprobe" in str(e).lower():
                    user_error_message += "وابستگی FFmpeg/FFprobe یافت نشد یا با خطا مواجه شد. "
                user_error_message += "مطمئن شوید فایل ویدیو دارای ترک صوتی است و فرمت آن پشتیبانی می‌شود."
                await client.edit_message(processing_msg, user_error_message)
                if not mime_type_attr.startswith('audio/'):
                    return
                actual_audio_file_to_process = download_path_initial

        final_audio_for_processing = None
        should_process_as_audio = (
            media_type_for_log.startswith("audio") or
            media_type_for_log.startswith("voice") or
            media_type_for_log.startswith("document_audio") or
            video_converted_to_audio_successfully
        )
        if not should_process_as_audio and mime_type_attr.startswith('audio/') and actual_audio_file_to_process == download_path_initial:
            should_process_as_audio = True

        if should_process_as_audio:
            standardized_audio_path_obj = TEMP_DIR / \
                f"{Path(actual_audio_file_to_process).stem}_standardized.{AUDIO_OUTPUT_FORMAT}"
            standardized_audio_path = str(standardized_audio_path_obj)
            try:
                logger.info(
                    f"Attempting to standardize audio file: {actual_audio_file_to_process}")
                audio_seg = AudioSegment.from_file(
                    actual_audio_file_to_process)
                if len(audio_seg) == 0:
                    logger.error(
                        f"Audio segment for standardization from '{actual_audio_file_to_process}' is empty.")
                    raise ValueError(
                        f"محتوای صوتی در فایل '{Path(actual_audio_file_to_process).name}' برای استانداردسازی خالی است.")

                audio_seg_standardized = audio_seg.set_channels(
                    1).set_frame_rate(AUDIO_SAMPLE_RATE)
                audio_seg_standardized.export(
                    standardized_audio_path, format=AUDIO_OUTPUT_FORMAT,
                    codec=AUDIO_OUTPUT_CODEC if AUDIO_OUTPUT_FORMAT == "ogg" else None,
                    bitrate=AUDIO_OUTPUT_BITRATE
                )
                if not standardized_audio_path_obj.exists() or standardized_audio_path_obj.stat().st_size < 256:
                    check_std_segment = AudioSegment.from_file(
                        standardized_audio_path)
                    if len(check_std_segment) == 0:
                        logger.error(
                            f"Standardized audio file '{standardized_audio_path}' is effectively empty.")
                        raise ValueError(
                            f"فایل صوتی استاندارد شده ('{standardized_audio_path_obj.name}') خالی یا نامعتبر است.")

                logger.info(
                    f"Audio standardized successfully to: {standardized_audio_path}, duration: {len(audio_seg_standardized)/1000.0:.2f}s")
                final_audio_for_processing = standardized_audio_path
                if standardized_audio_path != actual_audio_file_to_process:
                    local_files_to_cleanup_main_handler.append(
                        standardized_audio_path)
            except Exception as e:
                logger.warning(
                    f"Could not standardize audio '{actual_audio_file_to_process}': {e}. Proceeding with original/converted.", exc_info=True)
                await client.edit_message(processing_msg, processing_msg.text + f"\n⚠️ اخطار: نتوانستیم فایل صوتی را استاندارد کنیم. تلاش برای پردازش با فایل موجود... ({str(e)[:50]})")
                try:
                    fallback_segment = AudioSegment.from_file(
                        actual_audio_file_to_process)
                    if len(fallback_segment) == 0:
                        logger.error(
                            f"Fallback audio file '{actual_audio_file_to_process}' is empty.")
                        await client.edit_message(processing_msg, f"❌ فایل صوتی برای پردازش ({Path(actual_audio_file_to_process).name}) خالی است.")
                        return
                    final_audio_for_processing = actual_audio_file_to_process
                except Exception as fallback_e:
                    logger.error(
                        f"Fallback audio file '{actual_audio_file_to_process}' is also invalid: {fallback_e}", exc_info=True)
                    await client.edit_message(processing_msg, f"❌ فایل صوتی برای پردازش ({Path(actual_audio_file_to_process).name}) معتبر نیست: {fallback_e}")
                    return

            if final_audio_for_processing:
                await process_audio_file(event, final_audio_for_processing, original_name_base, chat_id, processing_msg)
            else:
                logger.error("No valid audio file determined for processing.")
                await client.edit_message(processing_msg, "❌ نتوانستیم فایل صوتی معتبری برای پردازش آماده کنیم.")
        else:
            # if not already handled by video fail
            if not (media_type_for_log.startswith("document_video") or mime_type_attr.startswith("video/")):
                logger.info(
                    f"File '{file_name_attr}' not processed as audio. Type: {media_type_for_log}, MIME: {mime_type_attr}")
                # This case should ideally be caught by initial unsupported file type check.

    except Exception as e:
        logger.exception(
            f"Unhandled error in handle_media_message for {file_name_attr}: {e}")
        try:
            await client.edit_message(processing_msg, f"❌ متأسفانه یک خطای ناشناخته در پردازش فایل شما رخ داد: {str(e)[:100]}")
        except:
            pass
    finally:
        unique_cleanup_paths_main = list(
            set(local_files_to_cleanup_main_handler))
        await cleanup_files_and_dirs(*unique_cleanup_paths_main)
        if TEMP_EXTRACTION_DIR.exists() and any(TEMP_EXTRACTION_DIR.iterdir()):
            logger.info(
                f"Cleaning up main extraction directory: {TEMP_EXTRACTION_DIR}")
            await cleanup_files_and_dirs(TEMP_EXTRACTION_DIR)
            TEMP_EXTRACTION_DIR.mkdir(parents=True, exist_ok=True)


async def main():
    global GDRIVE_SAVE_ENABLED
    logger.info("Starting LinguaScribe Bot...")
    for proxy_var in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
        if proxy_var in os.environ:
            logger.warning(f"Unsetting conflicting proxy environment variable: {proxy_var}")
            del os.environ[proxy_var]

    try:
        logger.info("Attempting to mount Google Drive...")
        drive.mount(GDRIVE_ROOT_PATH_STR, force_remount=True)
        if Path(GDRIVE_MYDRIVE_PATH_STR).exists():
            GDRIVE_STTCOLAB_PATH.mkdir(parents=True, exist_ok=True)
            GDRIVE_SAVE_ENABLED = True
            logger.info(f"Google Drive mounted successfully. Target directory: {GDRIVE_STTCOLAB_PATH_STR}")
        else:
            logger.error(
                f"Google Drive mount seemed to succeed, but MyDrive path ({GDRIVE_MYDRIVE_PATH_STR}) not found.")
            GDRIVE_SAVE_ENABLED = False
    except Exception as e:
        logger.error(
            f"Failed to mount Google Drive or create target directory: {e}. Files will NOT be saved to Drive.")
        GDRIVE_SAVE_ENABLED = False

    if TEMP_DIR.exists():
        try:
            shutil.rmtree(TEMP_DIR)  # Clean up before starting
            logger.info(f"Cleaned up old temp directory: {TEMP_DIR}")
        except Exception as e:
            logger.error(
                f"Error cleaning up temp directory {TEMP_DIR} at startup: {e}")
    TEMP_DIR.mkdir(parents=True, exist_ok=True)  # Recreate
    TEMP_EXTRACTION_DIR.mkdir(parents=True, exist_ok=True)  # Recreate

    await client.start(bot_token=BOT_TOKEN)
    me = await client.get_me()
    logger.info(f"Bot @{me.username} started successfully!")
    logger.info(f"Using {len(GOOGLE_API_KEYS_LIST)} Google API Key(s).")
    logger.info(
        f"Initial Google API Key: ...{_active_google_api_key[-4:] if _active_google_api_key else 'N/A'}")
    logger.info(
        f"Google Drive saving: {'ENABLED' if GDRIVE_SAVE_ENABLED else 'DISABLED'}")

    try:
        await client.run_until_disconnected()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user (Ctrl+C)")
    finally:
        await client.disconnect()
        logger.info("Bot disconnected.")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())