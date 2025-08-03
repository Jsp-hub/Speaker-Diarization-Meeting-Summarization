import os
import wave
import json
import argparse
import numpy as np
import torch
import torchaudio
from flask import Flask, request, jsonify

from diarization import SpeakerCNN, diarize_and_identify_array
from vosk import Model as VoskModel, KaldiRecognizer
from meetSummary import MeetingSummarizer

# ==== AUDIO CONVERSION ====
def convert_to_wav(src_path, dst_path="converted.wav"):
    os.system(f'ffmpeg -i "{src_path}" -ac 1 -ar 16000 "{dst_path}" -y -loglevel panic')
    return dst_path

# ==== DIARIZATION ====
def load_speaker_model(spd_path="spd.pth", spk2idx_path="spk2idx.pth"):
    spk2idx = torch.load(spk2idx_path)
    model = SpeakerCNN(len(spk2idx))
    model.load_state_dict(torch.load(spd_path, map_location="cpu"))
    model.eval()
    return model, spk2idx

def diarize_audio(wav_path, model, spk2idx):
    wav_tensor, sr = torchaudio.load(wav_path)
    wav = wav_tensor.squeeze(0).numpy()
    return diarize_and_identify_array(wav, sr, model, spk2idx)

# ==== TRANSCRIPTION ====
def transcribe_with_vosk(wav_path, vosk_model_dir="vosk-model-small-en-us-0.15"):
    wf = wave.open(wav_path, "rb")
    vosk_model = VoskModel(vosk_model_dir)
    rec = KaldiRecognizer(vosk_model, wf.getframerate())
    rec.SetWords(True)

    words = []
    while True:
        data = wf.readframes(4000)
        if not data:
            break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            words += res.get("result", [])
    final = json.loads(rec.FinalResult())
    words += final.get("result", [])
    return words

# ==== TAGGING ====
def tag_segments_with_text(segments, words):
    tagged = []
    for seg in segments:
        seg_words = [w["word"] for w in words if seg["start"] <= w["start"] < seg["end"]]
        if seg_words:
            tagged.append({
                "speaker": seg["speaker"],
                "start": seg["start"],
                "end": seg["end"],
                "text": " ".join(seg_words)
            })
    return tagged

# ==== SUMMARIZATION ====
def summarize_transcript(tagged_segments, summarizer_model="final-meeting-summarization-model"):
    transcript = "\n".join(f"{t['speaker']}: {t['text']}" for t in tagged_segments)
    summarizer = MeetingSummarizer(summarizer_model)
    return summarizer.summarize_transcript(transcript)

# ==== PIPELINE ====
def run_pipeline(input_audio_path):
    wav_path = convert_to_wav(input_audio_path)
    model, spk2idx = load_speaker_model()
    segments = diarize_audio(wav_path, model, spk2idx)
    words = transcribe_with_vosk(wav_path)
    tagged = tag_segments_with_text(segments, words)
    summary = summarize_transcript(tagged)
    return tagged, summary

# ==== FLASK API ====
app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio = request.files["audio"]
    audio_path = "uploaded_audio.wav"
    audio.save(audio_path)

    try:
        tagged, summary = run_pipeline(audio_path)
        return jsonify({
            "tagged_segments": tagged,
            "summary": summary
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==== CLI SUPPORT ====
def run_cli():
    parser = argparse.ArgumentParser(description="Run meeting summarization pipeline")
    parser.add_argument("audio_path", help="Path to the input audio file")
    args = parser.parse_args()

    tagged, summary = run_pipeline(args.audio_path)
    print("\n--- TAGGED SEGMENTS ---")
    for seg in tagged:
        print(f"{seg['speaker']} ({seg['start']:.2f}-{seg['end']:.2f}): {seg['text']}")

    print("\n--- SUMMARY ---")
    print(summary)

# ==== ENTRY POINT ====
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        app.run(host="0.0.0.0", port=5000)
    else:
        run_cli()
