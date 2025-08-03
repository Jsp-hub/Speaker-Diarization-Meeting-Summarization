import torch
from transformers import pipeline

def post_process_meeting_summary(text):
    return text.strip()

class MeetingSummarizer:
    def __init__(self, model_path):
        """Load your fine-tuned HuggingFace summarization model."""
        device = 0 if torch.cuda.is_available() else -1
        self.summarizer = pipeline(
            "summarization",
            model=model_path,
            tokenizer=model_path,
            device=device
        )

    def summarize_transcript(self, transcript, max_length = 180):
        """Chunk long transcripts, summarize, then post-process."""
        words = transcript.split()
        if len(words) > 1000:
            # split into 800-word chunks
            chunks = [
                " ".join(words[i:i+800])
                for i in range(0, len(words), 800)
            ]
            parts = []
            per_chunk_len = max_length // len(chunks)
            for chunk in chunks:
                out = self.summarizer(chunk,
                                      max_length=per_chunk_len,
                                      min_length=30,
                                      do_sample=False)
                parts.append(out[0]['summary_text'])
            full = " ".join(parts)
        else:
            out = self.summarizer(transcript,
                                  max_length=max_length,
                                  min_length=30,
                                  do_sample=False)
            full = out[0]['summary_text']
        return post_process_meeting_summary(full)