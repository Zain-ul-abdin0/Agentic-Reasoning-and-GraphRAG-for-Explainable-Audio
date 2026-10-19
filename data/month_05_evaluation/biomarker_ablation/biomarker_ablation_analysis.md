# Biomarker Contribution Analysis

## Purpose

This analysis estimates how individual biomarkers and feature-source groups affect screening performance against the PHQ-8 >= 10 reference label.

## Baseline

- Participants: 113
- Deployed matched-biomarker threshold: 6
- Deployed balanced accuracy: 0.726
- Deployed sensitivity: 0.732
- Deployed specificity: 0.719
- Optimized baseline threshold: 6
- Optimized baseline balanced accuracy: 0.726

## Most Important Biomarkers By Leave-One-Out

- High Pause Ratio: optimized balanced-accuracy delta -0.035; matched in 99 participants.
- Low Vocal Energy: optimized balanced-accuracy delta -0.027; matched in 21 participants.
- High MFCC 2 Variability: optimized balanced-accuracy delta -0.019; matched in 72 participants.
- Low Mean Pitch: optimized balanced-accuracy delta -0.018; matched in 13 participants.
- Low Speech Rate: optimized balanced-accuracy delta -0.018; matched in 101 participants.

## Best Single Biomarkers

- High Mean Pitch: balanced accuracy 0.656, sensitivity 0.732, specificity 0.579.
- High Pitch Variability: balanced accuracy 0.647, sensitivity 0.750, specificity 0.544.
- Low Vocal Energy: balanced accuracy 0.581, sensitivity 0.268, specificity 0.895.
- Low MFCC 3 Mean: balanced accuracy 0.577, sensitivity 0.804, specificity 0.351.
- High Pause Ratio: balanced accuracy 0.570, sensitivity 0.946, specificity 0.193.

## Feature-Source Group Ablation

- transcript_timing: optimized balanced-accuracy delta -0.051. Removed biomarkers: High Pause Ratio; Low Speech Rate.
- covarep_voice: optimized balanced-accuracy delta -0.043. Removed biomarkers: High Jitter; High Mean Pitch; High Pitch Variability; Low Mean Pitch.
- audio_waveform_spectral: optimized balanced-accuracy delta -0.025. Removed biomarkers: High MFCC 1 Mean; High MFCC 2 Variability; High MFCC 4 Variability; Low MFCC 3 Mean; Low Vocal Energy.

## Interpretation

Negative delta means performance decreased after removing that biomarker or group. A larger negative delta indicates stronger contribution to the screening rule.
