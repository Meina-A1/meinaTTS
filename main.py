import os
import random
import numpy as np
import torch
import sounddevice as sd
import soundfile as sf
from chatterbox.tts import ChatterboxTTS
from faster_whisper import WhisperModel
import threading
import json
import time
from collections import deque
import logging

# --- Kivy Imports ---
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.button import Button
from kivy.uix.spinner import Spinner
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.scrollview import ScrollView
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.textinput import TextInput

os.environ["KIVY_NO_CONSOLELOG"] = "1"  # Disable Kivy's console logs
logging.getLogger("faster_whisper").setLevel(logging.WARNING)
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# --- Constants and Configuration ---
CONFIG_FILE = "config.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_ASR_MODEL_SIZE = "base"
SAMPLE_RATE = 16000 
CHUNK_SIZE = 1024 # Process audio in chunks of 1024 samples
SILENCE_DURATION_S = 1.5 # Seconds of silence to end recording

# Supported languages for ASR
ASR_LANGUAGES = [
    'en', 'es', 'fr', 'de', 'it', 'ja', 'zh', 'ko', 'nl', 'ru', 
    'pt', 'tr', 'pl', 'ca', 'fa', 'uk', 'cs', 'ar', 'hu', 'fi'
]

# --- Configuration Management ---
def load_config():
    """Loads settings from the config file, providing defaults if it doesn't exist."""
    defaults = {
        "input_device": "Default",
        "output_device": "Default",
        "audio_prompt_path": "jps9.wav",
        "exaggeration": 1.0,
        "temperature": 1.0,
        "cfg_weight": 0.7,
        "seed": 0,
        "vad_threshold": 0.01, # Default sensitivity for VAD
        "asr_model_size": DEFAULT_ASR_MODEL_SIZE,
        "asr_language": "en"
    }
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            defaults.update(config)
    return defaults

def save_config(config):
    """Saves the current settings to the config file."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

# --- Core TTS and ASR Functions ---
def set_seed(seed: int):
    """Sets random seeds for reproducibility."""
    if seed != 0:
        torch.manual_seed(seed)
        # Add other seed settings as in the original script if needed

def generate_tts_audio(model, text, audio_prompt_path, exaggeration, temperature, seed_num, cfgw):
    """Generates audio using ChatterboxTTS."""
    set_seed(int(seed_num))
    if not os.path.exists(audio_prompt_path):
        raise FileNotFoundError(f"Audio prompt file not found: {audio_prompt_path}")
    wav = model.generate(
        text, audio_prompt_path=audio_prompt_path, exaggeration=exaggeration,
        temperature=temperature, cfg_weight=cfgw
    )
    return (model.sr, wav.squeeze(0).numpy())

# --- Kivy GUI Application ---
class VoiceChangerApp(App):
    def build(self):
        """Builds the Kivy interface."""
        self.title = "Real-Time Voice Changer (VAD Edition)"
        Window.size = (500, 800)  # Slightly taller for new controls
        self.config = load_config()
        self.tts_model = None
        self.asr_model = None
        self.is_running = False
        self.processing_thread = None

        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        settings_layout = GridLayout(cols=2, size_hint_y=None, height=350)  # Increased height

        # Device Selection
        self.input_devices, self.output_devices = self.get_audio_devices()
        settings_layout.add_widget(Label(text="Input Device:"))
        self.input_spinner = Spinner(text=self.config.get("input_device", "Default"), values=list(self.input_devices.keys()))
        self.input_spinner.bind(text=self.on_setting_change)
        settings_layout.add_widget(self.input_spinner)

        settings_layout.add_widget(Label(text="Output Device:"))
        self.output_spinner = Spinner(text=self.config.get("output_device", "Default"), values=list(self.output_devices.keys()))
        self.output_spinner.bind(text=self.on_setting_change)
        settings_layout.add_widget(self.output_spinner)
        
        settings_layout.add_widget(Label(text="Audio Prompt:"))
        self.prompt_button = Button(text=os.path.basename(self.config.get("audio_prompt_path", "Choose...")))
        self.prompt_button.bind(on_press=self.show_file_chooser)
        settings_layout.add_widget(self.prompt_button)

        # ASR Language Selection
        settings_layout.add_widget(Label(text="ASR Language:"))
        self.language_spinner = Spinner(
            text=self.config.get("asr_language", "en"),
            values=ASR_LANGUAGES
        )
        self.language_spinner.bind(text=self.on_setting_change)
        settings_layout.add_widget(self.language_spinner)

        # ASR Model Size Selection
        settings_layout.add_widget(Label(text="ASR Model Size:"))
        self.model_size_spinner = Spinner(
            text=self.config.get("asr_model_size", DEFAULT_ASR_MODEL_SIZE),
            values=["tiny", "base", "small", "medium", "large-v1", "large-v2"]
        )
        self.model_size_spinner.bind(text=self.on_setting_change)
        settings_layout.add_widget(self.model_size_spinner)

        # Sliders with text inputs
        self.vad_slider, self.vad_input = self.create_slider_with_input(
            "VAD Threshold:", 0.001, 0.1, "vad_threshold", settings_layout, "{:.3f}")
        self.exaggeration_slider, self.exaggeration_input = self.create_slider_with_input(
            "Exaggeration:", 0.0, 2.0, "exaggeration", settings_layout)
        self.temperature_slider, self.temperature_input = self.create_slider_with_input(
            "Temperature:", 0.1, 2.0, "temperature", settings_layout)
        self.cfg_slider, self.cfg_input = self.create_slider_with_input(
            "CFG Weight:", 0.0, 1.0, "cfg_weight", settings_layout)

        main_layout.add_widget(settings_layout)

        main_layout.add_widget(Label(text="Log", size_hint_y=None, height=20))
        self.log_scroll = ScrollView(size_hint=(1, 1))
        self.log_label = Label(text="", size_hint_y=None, markup=True)
        self.log_label.bind(texture_size=self.log_label.setter('size'))
        self.log_scroll.add_widget(self.log_label)
        main_layout.add_widget(self.log_scroll)
        
        self.control_button = Button(text="Start", size_hint_y=None, height=50, background_color=(0, 1, 0, 1))
        self.control_button.bind(on_press=self.toggle_processing)
        main_layout.add_widget(self.control_button)

        Clock.schedule_once(lambda dt: self.log_message(f"Welcome! Using device: [b]{DEVICE}[/b]"))
        Clock.schedule_once(lambda dt: self.log_message(f"Adjust VAD Threshold for your mic sensitivity."))
        
        return main_layout

    def create_slider_with_input(self, name, min_val, max_val, config_key, layout, fmt_str="{:.2f}"):
        """Creates a slider with a text input for direct value entry."""
        layout.add_widget(Label(text=name))
        slider_layout = BoxLayout(orientation='horizontal')
        
        # Create slider
        slider = Slider(min=min_val, max=max_val, value=self.config.get(config_key, (min_val + max_val) / 2))
        
        # Create text input
        text_input = TextInput(
            text=fmt_str.format(slider.value),
            size_hint_x=0.3,
            multiline=False,
            input_filter='float'
        )
        
        # Update functions
        def update_input(instance, value):
            text_input.text = fmt_str.format(value)
            
        def update_slider(instance):
            try:
                value = float(instance.text)
                if min_val <= value <= max_val:
                    slider.value = value
                else:
                    # Reset to current value if out of range
                    instance.text = fmt_str.format(slider.value)
            except ValueError:
                # Reset to current value if invalid input
                instance.text = fmt_str.format(slider.value)
        
        # Bindings
        slider.bind(value=update_input)
        slider.bind(value=lambda i, v: self.on_setting_change(i, v))
        slider.config_key = config_key
        
        text_input.bind(on_text_validate=update_slider)
        text_input.bind(focus=lambda instance, value: update_slider(instance) if not value else None)
        
        slider_layout.add_widget(slider)
        slider_layout.add_widget(text_input)
        layout.add_widget(slider_layout)
        
        # Set initial value
        update_input(slider, slider.value)
        
        return slider, text_input

    def get_audio_devices(self):
        try:
            devices = sd.query_devices()
            inputs = {"Default": None}
            outputs = {"Default": None}
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    inputs[dev['name']] = i
                if dev['max_output_channels'] > 0:
                    outputs[dev['name']] = i
            return inputs, outputs
        except Exception as e:
            return {"Default": None}, {"Default": None}

    def log_message(self, message):
        def _log(dt):
            self.log_label.text += f"{message}\n"
            self.log_scroll.scroll_y = 0
        Clock.schedule_once(_log)

    def on_setting_change(self, instance, value):
        if isinstance(instance, Spinner):
            if instance == self.input_spinner: 
                self.config["input_device"] = value
            elif instance == self.output_spinner: 
                self.config["output_device"] = value
            elif instance == self.language_spinner:
                self.config["asr_language"] = value
            elif instance == self.model_size_spinner:
                self.config["asr_model_size"] = value
        elif isinstance(instance, Slider):
            self.config[instance.config_key] = value
        elif isinstance(instance, str):
             self.config["audio_prompt_path"] = value
             self.prompt_button.text = os.path.basename(value)
        save_config(self.config)

    def show_file_chooser(self, instance):
        content = BoxLayout(orientation='vertical')
        filechooser = FileChooserListView(filters=['*.wav'])
        def select_file(fc_instance, selection, touch):
            if selection:
                self.on_setting_change("audio_prompt_path", selection[0])
                self.popup.dismiss()
        filechooser.bind(on_submit=select_file)
        content.add_widget(filechooser)
        self.popup = Popup(title="Choose an audio prompt (.wav)", content=content, size_hint=(0.9, 0.9))
        self.popup.open()

    def toggle_processing(self, instance):
        if self.is_running:
            self.is_running = False
            self.control_button.text = "Start"
            self.control_button.background_color = (0, 1, 0, 1)
            self.log_message(">>> Processing stopped by user.")
        else:
            self.is_running = True
            self.control_button.text = "Stop"
            self.control_button.background_color = (1, 0, 0, 1)
            self.log_message(">>> Starting processing...")
            self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
            self.processing_thread.start()

    def on_stop(self):
        self.is_running = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2)
    
    def processing_loop(self):
        try:
            # --- Load Models ---
            if self.tts_model is None:
                self.log_message("Loading TTS model (Chatterbox)...")
                self.tts_model = ChatterboxTTS.from_pretrained(DEVICE)
                self.log_message("TTS model loaded.")
                
            # Load ASR model if needed
            current_model_size = self.config.get("asr_model_size", DEFAULT_ASR_MODEL_SIZE)
            if (self.asr_model is None or 
                self.asr_model.model_size != current_model_size):
                self.log_message(f"Loading ASR model ({current_model_size})...")
                self.asr_model = WhisperModel(
                    current_model_size, 
                    device=DEVICE, 
                    compute_type="float16" if DEVICE == "cuda" else "int8"
                )
                self.log_message("ASR model loaded.")

            input_dev_idx = self.input_devices.get(self.config["input_device"])
            output_dev_idx = self.output_devices.get(self.config["output_device"])
            
            # VAD variables
            is_recording = False
            recorded_chunks = []
            silent_chunks = 0
            chunks_per_second = SAMPLE_RATE / CHUNK_SIZE
            silent_chunks_needed = int(SILENCE_DURATION_S * chunks_per_second)

            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', 
                                blocksize=CHUNK_SIZE, device=input_dev_idx) as stream:
                while self.is_running:
                    chunk, overflowed = stream.read(CHUNK_SIZE)
                    if overflowed:
                        self.log_message("Warning: Input overflowed")

                    # Simple RMS-based VAD
                    rms = np.sqrt(np.mean(chunk**2))

                    if is_recording:
                        recorded_chunks.append(chunk)
                        if rms < self.config["vad_threshold"]:
                            silent_chunks += 1
                        else: # Reset silence counter if speech is detected again
                            silent_chunks = 0

                        if silent_chunks > silent_chunks_needed:
                            self.log_message("Silence detected, processing...")
                            
                            # --- Process the recorded audio ---
                            full_recording = np.concatenate(recorded_chunks, axis=0).flatten()
                            
                            self.log_message("Transcribing...")
                            segments, _ = self.asr_model.transcribe(
                                full_recording, 
                                beam_size=5,
                                language=self.config["asr_language"]
                            )
                            transcribed_text = " ".join([seg.text for seg in segments]).strip()

                            if transcribed_text:
                                self.log_message(f"Input: [i]'{transcribed_text}'[/i]")
                                self.log_message("Generating TTS audio...")
                                tts_sr, tts_audio = generate_tts_audio(
                                    self.tts_model, transcribed_text,
                                    self.config["audio_prompt_path"], self.config["exaggeration"],
                                    self.config["temperature"], self.config["seed"], self.config["cfg_weight"]
                                )
                                self.log_message("Playing output...")
                                sd.play(tts_audio, samplerate=tts_sr, device=output_dev_idx)
                                sd.wait()
                            else:
                                self.log_message("Transcription was empty.")

                            # --- Reset for next utterance ---
                            self.log_message("---\nWaiting for speech...")
                            is_recording = False
                            recorded_chunks = []
                            silent_chunks = 0
                            
                    elif rms > self.config["vad_threshold"]:
                        self.log_message("Speech detected, recording...")
                        is_recording = True
                        silent_chunks = 0
                        recorded_chunks = [chunk] # Start with the chunk that triggered VAD
        
        except Exception as e:
            self.log_message(f"An error occurred in processing loop: {e}")
        
        # --- Cleanup on loop exit ---
        if self.is_running:
            Clock.schedule_once(lambda dt: self.toggle_processing(self.control_button))

if __name__ == "__main__":
    VoiceChangerApp().run()