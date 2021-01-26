"""
Various examples using the soundanalysis module.
"""

import sys
sys.path.append('..')
import soundanalysis
import settings

#filename = './Example_Textures/Bubbling_water.wav'
'''
allFilenames = ['Bubbling_water.wav',
                'Applause_-_enthusiastic2.wav',
                'Writing_with_pen_on_paper.wav',
                'white_noise_5s.wav']
filename = os.path.join('~/src/soundtextures/Example_Textures/',allFilenames[0])
'''
allFilenames = ['bubbles.wav']
filename = os.path.join(settings.SOUNDS_PATH,allFilenames[0])
#filename = './bubbles.wav'
#filename = './forest02.wav'

sana = soundanalysis.SoundAnalysis(soundfile=filename)

CASE = 0
if CASE == 0:
    plt.clf()
    sana.plot_spectrogram()
if CASE == 1:
    fbins = np.fft.fftfreq(len(sana.wave), 1/sana.samplingRate)
    ff = np.fft.fft(sana.wave)
    plt.clf()
    #plt.plot(fbins,np.log10(np.abs(ff)))
    plt.plot(fbins,np.abs(ff))
    plt.xlabel('Frequency')
    plt.ylabel('abs(FFT(x))')
if CASE == 2:
    fs = 20000
    fLims = [50,8000]
    fbank = FilterBank(6, fLims, 10*fs, fs)
    plt.clf()
    ax0=plt.subplot(2,1,1)
    fbank.plot()
    #plt.gca().set_xscale('log')
    plt.ylabel('Filter amplitude')
    plt.xlabel('Frequency')
    ax1=plt.subplot(2,1,2,sharex=ax0)
    #plt.plot(np.sum(fbank.filterHalfMat,axis=0))
    plt.plot(fbank.freqVec,np.sum(fbank.filtersTFpos**2,axis=0))
    plt.ylim([0,1.5])
    plt.show()
if CASE == 3:
    fbank = FilterBank(6, [50,8000], len(sana.wave), sana.samplingRate)
    #(fvec, fullTF) = fbank.get_transfer_function()
    fullTF = fbank.get_transfer_function()
    sana.apply_filterbank(fullTF)
    plt.clf()
    #plt.plot(sana.timeVec,np.real(sana.wave))
    #plt.plot(np.abs(sana.bandsFourier[5,:]))
    plt.plot(np.abs(sana.bandsFourier.T))
    #plt.plot(fullTF.T)
    plt.show()
if CASE == 4:
    fbank = FilterBank(6, [50,8000], len(sana.wave), sana.samplingRate)
    #(fvec, fullTF) = fbank.get_transfer_function()
    fullTF = fbank.get_transfer_function()
    sana.apply_filterbank(fullTF)
    #plt.clf()
    #plt.plot(sana.timeVec,np.real(sana.bands[0,:]))
    #plt.show()
    for indband in range(6):
        sana.play_band(indband, 3)  # Play only 3 seconds

if CASE == 5:
    fbank = FilterBank(12, [50,8000], len(sana.wave), sana.samplingRate)
    #(fvec, fullTF) = fbank.get_transfer_function()
    fullTF = fbank.get_transfer_function()
    sana.apply_filterbank(fullTF)
    downsampleFactor = 1
    sana.calculate_bands_envelopes(downsampleFactor=downsampleFactor)
    samplesToPlot = np.arange(5000)
    samplesToPlotEnv = np.arange(5000//downsampleFactor)
    plt.clf()
    fig, axs = plt.subplots(nrows=sana.nBands, ncols=1, sharex=True, num=1)
    for indband in range(sana.nBands):
        #axs[indband].subplot(sana.nBands,1,indband+1)
        axs[indband].plot(sana.timeVec[samplesToPlot],sana.bands[indband,samplesToPlot])
        axs[indband].plot(sana.bandsEnvelopesTimeVec[samplesToPlotEnv],
                          sana.bandsEnvelopes[indband,samplesToPlotEnv],'-')
    plt.show()

if CASE == 6:
    fbank = FilterBank(6, [20,8000], len(sana.wave), sana.samplingRate)
    #fbank = FilterBank(34, [20,10000], len(sana.wave), sana.samplingRate)
    #(fvec, fullTF) = fbank.get_transfer_function()
    fullTF = fbank.get_transfer_function()
    sana.apply_filterbank(fullTF)
    sana.calculate_bands_envelopes()
    plt.figure(1)
    sana.plot_bands_distributions()
    plt.figure(2)
    sana.apply_compression(0.3)
    sana.plot_bands_distributions()
    bandStats = sana.calculate_bands_stats()
    print(bandStats)
    sys.exit()

if CASE == 7:
    fbank = FilterBank(6, [20,8000], len(sana.wave), sana.samplingRate)
    #fbank = FilterBank(34, [20,10000], len(sana.wave), sana.samplingRate)
    #(fvec, fullTF) = fbank.get_transfer_function()
    fullTF = fbank.get_transfer_function()
    sana.apply_filterbank(fullTF)
    sana.calculate_bands_envelopes()
    sana.apply_compression(0.3)
    bandStats = sana.calculate_bands_stats()
    sana.plot_bands_stats()
    print(bandStats)
    sys.exit()

if CASE == 8:
    fbank = FilterBank(6, [20,8000], len(sana.wave), sana.samplingRate)
    fullTF = fbank.get_transfer_function()
    sana.apply_filterbank(fullTF)
    tvec,bands = sana.get_bands()
    bandsFFT = np.fft.fft(bands, axis=1)
    componentFFT = fullTF * bandsFFT
    components = np.fft.ifft(componentFFT, axis=1)
    newWave = np.real(np.sum(components, axis=0))
    errorWave = newWave-sana.wave
    samplesToPlot = np.arange(1000)
    plt.clf()
    #plt.plot(sana.wave[samplesToPlot],'.')
    #plt.plot(newWave[samplesToPlot],'o',mfc='none')
    plt.plot(errorWave)
    plt.show()
    # play_waveform(newWave, sana.samplingRate)

if CASE == 9:
    (soundStats, fbank) = sana.analyze(6, [20,8000])
    sys.exit()

if CASE == 10:
    ssyn = SoundSynthesis(6, [20,8000], len(sana.wave)//2, sana.samplingRate)
    soundStats = ssyn.calculate_stats()
    plt.figure(2)
    ssyn.soundObj.plot_bands_stats()
    plt.figure(1)
    ssyn.soundObj.plot_bands_distributions()
    #print(soundStats)

if CASE == 11:
    ssyn = SoundSynthesis(6, [20,8000], len(sana.wave)//2, sana.samplingRate)
    indband = 2
    statName = 'mean'
    newVar = 0.02
    newMean = 1.0
    soundStats = ssyn.calculate_stats()
    print('Band {} {} (before): {}'.format(indband, statName, soundStats[statName][2]))
    print('Band {} {} (before): {}'.format(indband, 'var', soundStats['var'][2]))
    ssyn.impose_oneband(indband, newMean, None)
    ssyn.soundObj.calculate_bands_envelopes()
    ssyn.soundObj.apply_compression()
    newSoundStats = ssyn.soundObj.calculate_bands_stats()
    print('Band {} {} (before): {}'.format(indband, statName, newSoundStats[statName][2]))
    print('Band {} {} (before): {}'.format(indband, 'var', soundStats['var'][2]))
    ssyn.soundObj.plot_bands_distributions()


CASE = 12
if CASE == 12:
    allFilenames = ['bubbles.wav','whitenoise.wav','pinknoise.wav','forest02.wav']
    filename = os.path.join('../jaranotebooks/soundsamples/',allFilenames[3])
    sana = SoundAnalysis(soundfile=filename)
    sana.analyze(6, [20,8000])
    #sana.plot_bands_distributions()
    #sana.plot_bands_stats()
    print(sana.bandsStats['skew'])
    print(sana.bandsStats['kurt']-3)

    #ac = np.correlate(sana.bandsEnvelopes[2],sana.bandsEnvelopes[2],'same')
    plt.clf()
    #plot_spectrogram(sana.bandsEnvelopes[2], sana.samplingRate, nfft=2048*4)
    for indband in range(sana.nBands):
        thisAx = plt.subplot(sana.nBands,1,sana.nBands-indband)
        plot_spectrum(sana.bandsEnvelopes[indband], sana.samplingRate, dc=False, maxFreq=100)
        plt.ylabel('Band {}'.format(indband))
        plt.xlim([-5,100])

plt.show()
