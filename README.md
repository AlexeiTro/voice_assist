# 🗣️ Voice Assist – Your Simple Voice Assistant

A lightweight and extensible voice assistant powered by Python. Recognize speech, execute custom voice commands, and make your life easier!
The goal hereby is to create a simple but effective voice assistant that can be plugged into phones to make life easier by letting the AI understand the callers intend.

## ✨ Features

- 🎤 **Speech Recognition** – Converts your voice into text using popular Python libraries.
- 🗂️ **Custom Commands** – Easily add your own commands and actions.
- 🛠️ **Extensible** – Plug in new features or connect to APIs.
- 🖥️ **Cross-Platform** – Works on Windows, macOS, and Linux (with Python 3.8+).

---

## 📦 Requirements

- Python 3.8+
- [`speech_recognition`](https://pypi.org/project/SpeechRecognition/)
- [`pyaudio`](https://pypi.org/project/PyAudio/) *(for microphone input)*
- [`uv`](https://github.com/astral-sh/uv) *(optional, for fast dependency management)*

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/AlexeiTro/voice_assist.git
cd voice_assist
```

## 📥 Install dependencies

With pip:

```bash
pip install -r requirements.txt
```

Or, for faster installs, with uv:

```bash
uv pip install -r requirements.txt
```

If you have issues with pyaudio installation, check PyAudio Installation Notes.

## ▶️ Usage

Simply run:

```bash
python main.py
```

The assistant will start listening. Try speaking a command!

## 🤖 Custom Commands

You can extend the assistant with your own commands!
See the commands/ directory (or relevant section in the code) for examples and instructions.

## 🤝 Contributing

Contributions are very welcome! (If you received an invitation)

Open issues for bugs or suggestions 🐛✨

Submit a pull request for features or improvements 🚀

## 📄 License
Apache License – see LICENSE for details.