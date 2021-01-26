"""
Example to show a spectrogram of a sound.
"""
import os
import sys
sys.path.append('..')
import soundanalysis
import settings
from matplotlib import pyplot as plt

filename = os.path.join(settings.SOUNDS_PATH,'bubbles.wav')

sana = soundanalysis.SoundAnalysis(soundfile=filename)

sana.plot_spectrogram()
plt.show()