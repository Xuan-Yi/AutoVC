import os
import pickle
import numpy as np
import soundfile as sf
import librosa
from scipy import signal
from scipy.signal import get_window
from librosa.filters import mel
from librosa.util import normalize
from numpy.random import RandomState
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--rootDir", dest="rootDir", default='./wavs', help="rootDir")
parser.add_argument("--targetDir", dest="targetDir", default='./spmel', help="spmel")

args = parser.parse_args()


# audio file directory
rootDir = args.rootDir
# spectrogram directory
targetDir = args.targetDir
dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)


def mel_gan_handler(x, fft_length = 1024, hop_length = 256,sr = 22050):
    wav = normalize(x)
    p = (fft_length - hop_length) // 2
    wav = np.squeeze(np.pad(wav, (p, p), "reflect"))
    fft = librosa.stft(
                       wav, 
                       n_fft = fft_length, 
                       hop_length = hop_length,
                       window = 'hann',
                       center = False
                     )
    # fft is complex here
    mag = abs(fft)
    mel_basis = mel(sr=sr, n_fft=1024, fmin = 0.0 , fmax=None, n_mels=80)
    mel_output = np.dot(mel_basis,mag)
    log_mel_spec = np.log10(np.maximum(1e-5,mel_output)).astype(np.float32)
    return log_mel_spec
    
# resample to 22.05 KHz 
new_rate = 22050
for subdir in sorted(subdirList):
    if not os.path.exists(os.path.join(targetDir, subdir)):
        os.makedirs(os.path.join(targetDir, subdir))
    _,_, fileList = next(os.walk(os.path.join(dirName,subdir)))
    for fileName in sorted(fileList):
        x, fs = sf.read(os.path.join(dirName,subdir,fileName))
        # change sample rate from 48000 to 22050, since mel_gan use 22050
        x = librosa.resample(x,orig_sr=fs,target_sr=new_rate)
        S = mel_gan_handler(x)
        S = S.transpose()
        np.save(os.path.join(targetDir, subdir, fileName[:-4]),
                 S.astype(np.float32), allow_pickle=False)
    print(f"Done --- {subdir}")