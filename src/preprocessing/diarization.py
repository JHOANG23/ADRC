"""
Author: Dr. Michelle Dana Cohn
Modified by: Joseph Hoang
"""

import os
import glob
import torch
import pandas as pd
from pyannote.audio import Pipeline
from textgrid import TextGrid, IntervalTier, Interval
from src.config import EXCEL_HOMETOWN_PATH, AUDIO_INPUT_PATH, DIARIZATION_PATH

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=os.getenv("HF_TOKEN"))

# send pipeline to GPU (when available)
pipeline.to(torch.device("cuda"))
sheet = pd.read_excel(EXCEL_HOMETOWN_PATH, sheet_name=1, usecols='A') # Comment out for now
file_names = sheet.iloc[:,0].str.removesuffix(".wav")
# loop through all wav files from the spanish recordings in the hometown dataset
for pid in file_names:
    matches = glob.glob(f"{AUDIO_INPUT_PATH}/{pid}*.wav")
    if not matches:
        continue

    full_wav_path = matches[0]
    file_name = os.path.basename(full_wav_path)
    file_name = os.path.splitext(file_name)[0]

    txt_output_file = f"{DIARIZATION_PATH}/{file_name}_diarization_output.txt"
    tg_output_file = f"{DIARIZATION_PATH}/{file_name}_diarization_output.TextGrid"

    # create diarized .txt file
    out = pipeline(full_wav_path)
    diarization = out.speaker_diarization
    with open(txt_output_file, "w") as f:
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            line = f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}\n"
            f.write(line)

    #Create diarized .TextGrid files
    tg = TextGrid()
    segment_extent = diarization.get_timeline().extent()
    tg.maxTime = segment_extent.end

    for speaker in diarization.labels():
        tier = IntervalTier(name=f"speaker_{speaker}", maxTime=tg.maxTime)
        speaker_segments = diarization.label_timeline(speaker)

        for segment in speaker_segments:
            interval = Interval(segment.start, segment.end, f"speaker_{speaker}")
            tier.addInterval(interval)
        
        tg.append(tier)
    tg.write(tg_output_file)

    print(f"Diarization exported to {txt_output_file} and {tg_output_file}")