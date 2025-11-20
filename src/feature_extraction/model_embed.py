# Models to account for:  XLSR-53, wav2vec, whisper
import os
import time
import torch
import numpy as np
import pandas as pd
from src.utils import split_wav
from transformers import Wav2Vec2FeatureExtractor, AutoModel

def _embed_audio(wav_arr, model, processor, device): 
    inputs = processor(
        wav_arr, 
        sampling_rate = 16000, 
        return_tensors='pt', 
        padding=True 
    ).to(device) 
    
    with torch.no_grad(): 
        embedding_output = model(**inputs) 
        hidden = embedding_output.last_hidden_state #Shape: (batch_size=1, seq_length, hidden_size) 
    mean_pooled_output = hidden.mean(dim=1) #Shape: (batch_size=1, hidden_size) 
    return mean_pooled_output.cpu()

def embed_audio(df, input_dir, model_path, mode='equal', N=4, batch_size=1):
    start_time = time.time()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.to(device)

    embeddings = {'en': [], 'es': []}
    labels = {'en': [], 'es': []}

    languages = ['en', 'es']
    for lang in languages:
        for row in df[df['language'] == lang].itertuples(index=False):
            if not row.file_name.endswith(".wav") or row.synd2 == -1.0 or np.isnan(row.synd2):
                continue
            print(f"Embedding: {row.file_name}")
            wav_file = os.path.join(input_dir, row.language, row.file_name)
            file_kb = os.path.getsize(wav_file) / 1024
            if file_kb > 25000:        # 20,000 KB = 20MB
                N = 5
            else:
                N = 1
            segments, sr = split_wav(wav_file, mode, N)
            for i, seg in enumerate(segments):
                embed = _embed_audio([seg], model, processor, device)
                embeddings[lang].append(embed)
                labels[lang].append(row.synd2)
                
    X_en_tensor = torch.cat(embeddings['en'], dim=0)
    Y_en_tensor = torch.tensor(labels['en'], dtype=torch.float32)
    X_es_tensor = torch.cat(embeddings['es'], dim=0)
    Y_es_tensor = torch.tensor(labels['es'], dtype=torch.float32)
    
    end_time = time.time()
    print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")

    return X_en_tensor, Y_en_tensor, X_es_tensor, Y_es_tensor

def main():
    MODEL_PATH = '/home/jobe/models/wav2vec2-large-xlsr-53'
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_PATH)
    model_gpu = AutoModel.from_pretrained(MODEL_PATH).to(device)
    model_cpu = AutoModel.from_pretrained(MODEL_PATH).to("cpu")
    wav_path = '../dataset/TrimmedWavFiles/en/56028trt.9.10.10_trimmed.wav'
    segments, sr = split_wav(wav_path, 'equal', 1)
    
    embeds = _embed_audio(segments, model_gpu, model_cpu, processor, device)

if __name__ == "__main__":
    main()