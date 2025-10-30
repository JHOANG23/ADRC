"""
This script takes in Praat .TextGrid files to identify 'interviewer' intervals within the audio files
to trim out. Once all intervals are trimmed, all audio splits are then combined into one .wav file. 
"""

import os
from textgrid import TextGrid
from pydub import AudioSegment

NUM_INITIAL_PAUSES_TO_SKIP = 2  # skip this many initial pauses

def trim_audio(wav_file_path, diarized_file_path, output_path, base_name):
    tg = TextGrid.fromFile(diarized_file_path)
    interviewer_tier = None
    patient_tier = None
    for tier in tg.tiers:
        if "SPEAKER_00" in tier.name:
            patient_tier = tier
        else:
            interviewer_tier = tier
    if interviewer_tier is None or patient_tier is None:
        raise ValueError("Missing one or both tiers (SPEAKER_00 / SPEAKER_01)")

    audio = AudioSegment.from_wav(wav_file_path)
    skip_until = 0.0
    pause_count = 0
    found_first_speech = False

    for interval in patient_tier.intervals:
        if interval.mark.strip():  # patient speech
            if not found_first_speech:
                found_first_speech = True
            continue
        # silence after first speech
        if found_first_speech and not interval.mark.strip():
            pause_count += 1
            if pause_count <= NUM_INITIAL_PAUSES_TO_SKIP:
                skip_until = interval.maxTime
            else:
                break
    # If no speech found, fallback to 0
    if skip_until == 0.0:
        for interval in patient_tier.intervals:
            if interval.mark.strip():
                skip_until = interval.minTime
                break

    # --- STEP 2: Single-pass interviewer-cutout logic ---
    segments_to_keep = []
    last_end = skip_until

    for interval in interviewer_tier.intervals:
        if interval.mark.strip():  # interviewer speaking
            start = interval.minTime
            end = interval.maxTime
            if start > last_end:
                segments_to_keep.append((last_end, start))
            last_end = max(last_end, end)

    # After last interviewer interval, keep remaining audio
    if last_end < audio.duration_seconds:
        segments_to_keep.append((last_end, audio.duration_seconds))

    # combine split audio
    segments_list = [
        audio[start * 1000 : end * 1000]
        for start, end in segments_to_keep
    ]
    output_audio = sum(segments_list)

    os.makedirs(output_path, exist_ok=True)
    out_path = os.path.join(output_path, f"{base_name}_trimmed.wav")
    output_audio.export(out_path, format="wav")

def main():
    print("Beginning trimming process")
    audio_input_path = '../dataset/WavFiles'
    diarization_path = '../dataset/SpeakerDiarization_Annotations'
    trimmed_output_path = '../dataset/TrimmedWavFiles'

    for file_name in os.listdir(diarization_path):
        if not file_name.endswith('.TextGrid'):
            continue
        wav_file_path = os.path.join(audio_input_path, file_name.replace("_diarization_output.TextGrid", ".wav"))
        diarized_file_path = os.path.join(diarization_path, file_name)
        base_name = file_name.removesuffix("_diarization_output.TextGrid")
        print(f"Trimming: {base_name}.wav")
        trim_audio(wav_file_path, diarized_file_path, trimmed_output_path, base_name)
    
    print("Finished trimming patient audio recordings.")

if __name__ == "__main__":
    main()
