########################################################################################
# Author: Dr. Michelle Dana Cohn
# Edited By: Joseph Hoang
# Prosodic analysis toolkit! v2
#
# Loops over individual files and measures prosodic features 
# Outputs: .csv file 
# 
#  MEASUREMENTS: 
#
# (1) Duration
# (2) F0 (sampled at 10 equidistant intervals)
# (3) Intensity (avg. over utterance)
# (4) Speech rate
# (5) Articulation rate
# (6) Jitter
# (7) Shimmer

#  REQUIREMENTS:
#  (1) .wav file 
# 
# MC 3/15/2023; updated 9/8/25
########################################################################################

import os
import torch
import numpy as np
import pandas as pd
import parselmouth
from textgrid import TextGrid
from parselmouth.praat import call
from src.config import TRIMMED_AUDIO_PATH, FEATURE_PATH, DIARIZATION_PATH

def interpolate_f0_intervals(df, num_intervals=10):
    interval_cols = [f"Mean_f0_Interval_{i}" for i in range(1, num_intervals + 1)]
    df[interval_cols] = df[interval_cols].interpolate(axis=1, limit_direction="both")
    return df

def compute_speech_rate_features(wav_file, diarized_file, voicedcount, snd):
    NUM_INITIAL_PAUSES_TO_SKIP=2
    snd = parselmouth.Sound(wav_file)
    total_duration = snd.get_total_duration()

    tg = TextGrid.fromFile(diarized_file)

    # identify patient speaker by longest interval
    durations = {
        tier.name: sum(interval.maxTime - interval.minTime for interval in tier.intervals if interval.mark.strip())
        for tier in tg.tiers
    }
    patient_tier_name = max(durations, key=durations.get)
    patient_tier = next(t for t in tg.tiers if t.name == patient_tier_name)

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
    # Collect patient speech intervals after skipping initial pauses
    patient_intervals = [
        (max(interval.minTime, skip_until), interval.maxTime)
        for interval in patient_tier.intervals
        if interval.mark.strip() and interval.maxTime > skip_until
    ]

    # Compute npause and phonation time
    npause = 0
    phonation_time = 0.0
    for interval in patient_intervals:
        phonation_time += interval[1] - interval[0]

    # Count pauses as gaps between consecutive intervals
    for i in range(1, len(patient_intervals)):
        gap = patient_intervals[i][0] - patient_intervals[i - 1][1]
        if gap > 0:
            npause += 1


    nsyllables = voicedcount  
    speakingrate = nsyllables / total_duration if total_duration > 0 else 0
    articulationrate = nsyllables / phonation_time if phonation_time > 0 else 0
    asd = phonation_time / max(nsyllables, 1)  # avoid divide by zero

    return {
        "npause": npause,
        "speakingrate": speakingrate,
        "articulationrate": articulationrate,
        "asd": asd,
        "voicedcount": voicedcount,
    }

def extract_prosodic_features(input_df: pd.DataFrame, output_dir: str, num_intervals: int = 10):
    """
    Measure prosodic features for all .wav files in a directory.
    Replicates the original Praat script functionality.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Initialize output table
    columns = [
        "Duration", "Intensity", "RMS",
        "Jitter_ABS", "Jitter_RAP", "Shimmer_Local", "Shimmer_dB",
        "LTAS_1_to_3_kHz_Pa", "LTAS_1_to_3_kHz_dB", "Mean_f0", "Sd_f0", "Max_F0",
        "voicedcount", "npause", "speakingrate", "articulationrate", "asd"
    ]
    columns += [f"Mean_f0_Interval_{i+1}" for i in range(num_intervals)]

    results = []
    f0_min, f0_max = 78, 350
    Y = []
    es_count = 0
    for row in input_df.itertuples(index=False):
        if not row.file_name.endswith(".wav") or row.synd2 == -1.0 or np.isnan(row.synd2):
            continue
        if row.language == 'es':
            count +=1
        print(f"Extracting feature from {row.file_name}")
        wav_path = os.path.join(TRIMMED_AUDIO_PATH, row.language, row.file_name)
        snd = parselmouth.Sound(wav_path)

        start = 0.0
        end = snd.get_total_duration()
        utterance_duration = end - start

        # Skip too-short audio
        if utterance_duration <= 0.040:
            continue

        # Intensity and RMS
        intensity_obj = call(snd, "To Intensity", 75, 0.0, True)
        mean_intensity = call(intensity_obj, "Get mean", 0, 0, "energy")
        rms = snd.get_rms()  # root-mean-square amplitude

        # Pitch features
        pitch = call(snd, "To Pitch", 0.0, f0_min, f0_max)
        overall_mean_f0 = call(pitch, "Get mean", 0, 0, "Hertz")
        overall_sd_f0 = call(pitch, "Get standard deviation", 0, 0, "Hertz")
        max_f0 = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")

        # Compute per-interval means
        interval_len = utterance_duration / num_intervals
        interval_means = []
        for i in range(num_intervals):
            interval_start = start + i * interval_len
            interval_end = interval_start + interval_len
            sub_mean = call(pitch, "Get mean", interval_start, interval_end, "Hertz")
            interval_means.append(sub_mean)

        # Voice perturbation measures (Jitter, Shimmer)
        point_proc = call([snd, pitch], "To PointProcess (cc)")
        jitter_abs = call(point_proc, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_rap = call(point_proc, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer_local = call([snd, point_proc], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_dB = call([snd, point_proc], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        # LTAS (Long-term average spectrum)
        ltas = call(snd, "To Ltas", 100.0)
        ltas_1to3_pa = call(ltas, "Get mean", 1000, 3000, "energy")
        ltas_1to3_db = call(ltas, "Get mean", 1000, 3000, "dB")

        # Placeholder speech rate measures (requires TextGrid or voiced segments)
        voicedcount = call(pitch, "Count voiced frames")
        base_name = row.file_name.removesuffix('_trimmed.wav')
        diarized_path = os.path.join(DIARIZATION_PATH, f"{base_name}_diarization_output.TextGrid")
        speech_rate_measures = compute_speech_rate_features(wav_path, diarized_path, voicedcount, snd)

        npause = speech_rate_measures['npause']  
        speakingrate = speech_rate_measures['speakingrate']  
        articulationrate = speech_rate_measures['articulationrate']  
        asd = speech_rate_measures['asd']  

        results.append([
            utterance_duration, mean_intensity, rms,
            jitter_abs, jitter_rap, shimmer_local, shimmer_dB,
            ltas_1to3_pa, ltas_1to3_db, overall_mean_f0, overall_sd_f0, max_f0,
            voicedcount, npause, speakingrate, articulationrate, asd,
            *interval_means
        ])
        Y.append(row.synd2)
    Y = np.array(Y)
    X_df = pd.DataFrame(results, columns=columns)
    X_df = interpolate_f0_intervals(X_df)
    X_tensor = torch.tensor(X_df.values, dtype=torch.float32).cpu()
    Y_tensor = torch.from_numpy(Y).long().cpu()

    X_en_tensor = X_tensor[:-es_count]
    Y_en_tensor = Y_tensor[:-es_count]
    X_es_tensor = X_tensor[-es_count:]
    Y_es_tensor = Y_tensor[-es_count:]

    print(f"Features extracted")

    return X_en_tensor, Y_en_tensor, X_es_tensor, Y_es_tensor
