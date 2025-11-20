"""
This script takes in Praat .TextGrid files to identify 'interviewer' intervals within the audio files
to trim out. Once all intervals are trimmed, all audio splits are then combined into one .wav file. 
"""

import os
import glob
import shutil
import pandas as pd
from textgrid import TextGrid
from pydub import AudioSegment
from src.config import  AUDIO_INPUT_PATH, DIARIZATION_PATH, TRIMMED_AUDIO_PATH

NUM_INITIAL_PAUSES_TO_SKIP = 2  # skip this many initial pauses

def trim_audio(wav_file_path, diarized_file_path, output_path, base_name):
    tg = TextGrid.fromFile(diarized_file_path)

    # identify patient speaker by longest interval
    durations = {
        tier.name: sum(interval.maxTime - interval.minTime for interval in tier.intervals if interval.mark.strip())
        for tier in tg.tiers
    }
    patient_tier_name = max(durations, key=durations.get)
    patient_tier = next(t for t in tg.tiers if t.name == patient_tier_name)
    interviewer_tiers = [t for t in tg.tiers if t.name != patient_tier_name]

    # find skip start time ---
    audio = AudioSegment.from_wav(wav_file_path)
    skip_until = 0.0
    pause_count = 0
    found_first_speech = False

    for interval in patient_tier.intervals:
        if interval.mark.strip():
            if not found_first_speech:
                found_first_speech = True
            continue
        if found_first_speech and not interval.mark.strip():
            pause_count += 1
            if pause_count <= NUM_INITIAL_PAUSES_TO_SKIP:
                skip_until = interval.maxTime
            else:
                break

    if skip_until == 0.0:
        for interval in patient_tier.intervals:
            if interval.mark.strip():
                skip_until = interval.minTime
                break

    # merge all interviewer intervals
    interviewer_intervals = []
    for tier in interviewer_tiers:
        for interval in tier.intervals:
            if interval.mark.strip():  # interviewer speaking
                interviewer_intervals.append((interval.minTime, interval.maxTime))
    interviewer_intervals.sort(key=lambda x: x[0])

    # determine segments to keep
    segments_to_keep = []
    last_end = skip_until

    for start, end in interviewer_intervals:
        if start > last_end:
            segments_to_keep.append((last_end, start))
        last_end = max(last_end, end)

    if last_end < audio.duration_seconds:
        segments_to_keep.append((last_end, audio.duration_seconds))

    # combine split audio segments
    segments_list = [audio[start * 1000 : end * 1000] for start, end in segments_to_keep]
    output_audio = sum(segments_list)
    output_audio = output_audio.set_frame_rate(16000)
    out_path = os.path.join(output_path, f"{base_name}_trimmed.wav")
    output_audio.export(out_path, format="wav")

def main():
    os.makedirs(AUDIO_INPUT_PATH, exist_ok=True)
    os.makedirs(DIARIZATION_PATH, exist_ok=True)
    os.makedirs(TRIMMED_AUDIO_PATH, exist_ok=True)
    os.makedirs(os.path.join(TRIMMED_AUDIO_PATH, "es"), exist_ok=True)
    os.makedirs(os.path.join(TRIMMED_AUDIO_PATH, "en"), exist_ok=True)

    print("Beginning trimming process")
    excel_file = '/home/jobe/datasets/ADReSS-M/combined_csvs.xlsx'
    spanish_sheet = pd.read_excel(excel_file, sheet_name=1, usecols='A')
    spanish_ids = set(spanish_sheet.iloc[:,0].str.removesuffix(".wav"))
    for file_name in os.listdir(DIARIZATION_PATH):
        if not file_name.endswith('.TextGrid'):
            continue
        wav_file_path = os.path.join(AUDIO_INPUT_PATH, file_name.replace("_diarization_output.TextGrid", ".wav"))
        diarized_file_path = os.path.join(DIARIZATION_PATH, file_name)
        base_name = file_name.split('.', 1)[0]

        if base_name in spanish_ids:
            output_path = os.path.join(TRIMMED_AUDIO_PATH, "es")
        else:
            output_path = os.path.join(TRIMMED_AUDIO_PATH, "en")

        base_name = file_name.removesuffix("_diarization_output.TextGrid")
        print(f"Trimming: {base_name}.wav")
        trim_audio(wav_file_path, diarized_file_path, output_path, base_name)
    print("Finished trimming patient audio recordings.")

if __name__ == "__main__":
    main()
