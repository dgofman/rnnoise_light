**Lightweight Noise Suppression for Audio Processing**

`rnnoise_light` is a streamlined, minimalistic version of the [Xiph RNNoise](https://github.com/xiph/rnnoise) library, designed for efficient real-time noise suppression on embedded systems, desktop UIs, or constrained environments.

Unlike the full RNNoise implementation, this version reduces dependencies and computational overhead while retaining usable audio quality. For integration into Flutter or real-time streaming systems, see my other project: [dgofman/audiostreamer](https://github.com/dgofman/audiostreamer).

---

### 🔊 Test Audio Samples

- [🌐 index.html](https://dgofman.github.io/rnnoise_light)
- [▶️ test_input.wav](https://dgofman.github.io/rnnoise_light/test_input.wav)
- [▶️ test_output.wav](https://dgofman.github.io/rnnoise_light/test_output.wav)
- [▶️ test_full_output.wav](https://dgofman.github.io/rnnoise_light/test_full_output.wav)
---


## 🔍 What is RNN?

**RNN (Recurrent Neural Network)** is a type of artificial neural network designed for processing sequential data such as audio. It uses memory of previous inputs to inform future predictions, making it suitable for speech and noise modeling.

---

## 🎧 Test Setup

### 1. Input Audio

- **Sample from YouTube:**  
  https://www.youtube.com/shorts/nRLKJq1eF0Q

- **Dataset:**  
  [Google's Noisy Speech Commands Dataset](https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz)  
  Includes samples like:
  ```
  speech_commands_v0.02/_background_noise_/running_tap.wav
  ```

### 2. Record Test Audio with Background Noise

Use `ffmpeg` to record your microphone with injected background noise:

```bash
ffmpeg -f dshow -i audio="{{MICROPHONE NAME}}" -t 13 -filter:a "volume=1.0" -ar 48000 -ac 1 test_input.wav
```

To list your available microphone devices:

```bash
ffmpeg -list_devices true -f dshow -i dummy
```

---

## ⚙️ Build Instructions

### Step 1: Compile

```bash
make
```

### Step 2: Run Noise Suppression

```bash
examples/rnnoise_demo.exe docs/test_input.wav docs/test_output.raw
```

### Step 3: Convert Output to WAV

```bash
ffmpeg -f s16le -ar 48000 -ac 1 -i docs/test_output.raw docs/test_output.wav
```

---

## 💡 Notes

- The `test_output.raw` is 16-bit PCM (mono, 48 kHz).
- Noise suppression is performed using a lightweight RNN-based model.
- Reduced `rnnoise_tables` can be used for performance experiments.
