#!/usr/bin/env python
#by wujian@2017.4.15

"""transform spectrogram to waveform"""

import sys
import wave
import numpy as np
sys.path.append("../LSTM_PIT")
from io_funcs import wave_io
import argparse

def main(argv):
  wnd_len = FLAGS.frame_len; #ms
  wnd_shift = FLAGS.frame_shift #ms
  fft_len = FLAGS.fft_len; # for 8k sample rate
  pre_em = FLAGS.pre_em
  
  spect_enhance = np.load(argv[1])  #load the enhanced stft features 
  wave_warpper = wave_io.WaveWrapper(argv[2],time_wnd=wnd_len, time_off=wnd_shift)
  wave_reconst = wave.open(argv[3], "wb")
  
  wnd_size = wave_warpper.get_wnd_size()
  wnd_rate = wave_warpper.get_wnd_rate()
  
  real_ifft = np.fft.irfft
  
  ham_wnd = np.hamming(wnd_size+1) #simulate the matlab hamming(N, 'periodic')
  ham_wnd = np.sqrt(ham_wnd[0:-1]);
  stride = range(0,wnd_size,wnd_rate)
  ham_wnd = ham_wnd/np.sqrt(np.sum(ham_wnd[stride]*ham_wnd[stride])) #nomilize the window
  spect_rows, spect_cols = spect_enhance.shape
  assert wave_warpper.get_frames_num() == spect_rows
  index = 0
  spect = np.zeros(spect_cols)
  reconst_pool = np.zeros((spect_rows - 1) * wnd_rate + wnd_size)
  for phase in wave_warpper.next_frame_phase(fft_len=fft_len,pre_em=pre_em):
      # exclude energy
      #spect[1: ] = np.sqrt(np.exp(spect_enhance[index][1: ]))
      spect= spect_enhance[index]
      reconst_pool[index * wnd_rate: index * wnd_rate + wnd_size] += \
                  real_ifft(spect * phase)[: wnd_size] * ham_wnd
      index += 1
  # remove pre-emphasis
  if pre_em.lower() == 'true':
    for x in range(1, reconst_pool.size):
      reconst_pool[x] += 0.97 * reconst_pool[x - 1]
  reconst_pool = reconst_pool / np.max(np.abs(reconst_pool)) * wave_warpper.get_upper_bound()
  
  wave_reconst.setnchannels(1)
  wave_reconst.setnframes(reconst_pool.size)
  wave_reconst.setsampwidth(2)
  wave_reconst.setframerate(wave_warpper.get_sample_rate())
  wave_reconst.writeframes(np.array(reconst_pool, dtype=np.int16).tostring())
  wave_reconst.close()
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--frame_len', type=int, default=32,
                      help='frame len in ms (int, default=32)')
  parser.add_argument('--frame_shift', type=int, default=16,
                      help='Frame shift in ms (int, default=16)')
  parser.add_argument('--fft_len', type=int, default=256,
                      help='FFT len in sample (int, default=256')
  parser.add_argument('--pre_em', type=str, default='False',
                      help='Pre-emphasis or not (str, default=Fasle)')
  FLAGS, unparsed = parser.parse_known_args()
  if len(unparsed) != 3:
    print "format error: %s [enhanced-npz-name] [origin-wave-name] [reconst-wave-name]" % sys.argv[0]
    sys.exit(1)

  main([sys.argv[0]] + unparsed)




