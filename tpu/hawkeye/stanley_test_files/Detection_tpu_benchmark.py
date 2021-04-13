import numpy as np
import librosa
import matplotlib.pyplot as plt
import wave
from pyaudio import PyAudio, paInt16
import os
#import pandas as pd
from sklearn import metrics
import time


class_list = {0: 'Cry', 1: 'Screaming', 2: 'Shatter', 3: 'Bark', 4: 'Boiling', 5: 'Toilet flush',
              6: 'Doorbell', 7: 'Telephone ring', 8: 'Water tap', 9: 'Blender', 10: 'Cupboard / drawer',
              11: 'Cupboard / drawer', 12: 'Cough', 13: 'Snoring', 14: 'Walk', 15: 'Speech',
              16: 'Alarm', 17: 'Cook', 18: 'Groan', 19: 'Sneeze', 20: 'Fall down', 99: 'no_evevt'}

class_list_file = {0: 'Cry', 1: 'Screaming', 2: 'Shatter', 3: 'Bark', 4: 'Boiling', 5: 'Toilet',
              6: 'Doorbell', 7: 'Telephone', 8: 'Water', 9: 'Blender', 10: 'Cupboard',
              11: 'Cupboard', 12: 'Cough', 13: 'Snoring', 14: 'Walk', 15: 'Speech',
              16: 'Alarm', 17: 'Cook', 18: 'Groan', 19: 'Sneeze', 20: 'Fall', 99: 'no_evevt'}


#def detection(model, inputs, sample_rate=48000, uni_sample=48000):
def detection_input (inputs, sample_rate=48000, uni_sample=48000):
    """Load 2-sec sampling to model for detection.

    Args:
        model: detection model.
        inputs: 2-sec sampling.
        sample_rate: the sampling rate of inputs.
        class_list: prediction index to classes

    Returns:
        if there is an event happen : print event class and probability
        or print 'no event'
    """

    def extract(x, sample_rate):
        """Transform the given signal into a logmel feature vector.

        Args:
        x (np.ndarray): Input time-series signal.
        sample_rate (number): Sampling rate of signal.

        Returns:
        np.ndarray: The logmel feature vector.
        """
        # Resample to target sampling rate
        x = librosa.resample(x, sample_rate, uni_sample)

        # Compute short-time Fourier transform
        D = librosa.stft(x, n_fft=1024, hop_length=512)
        '''
        print ('py extract, librosa.stft.dtype=', D.dtype)
        print ('shape=', D.shape)
        print ('output D=')
        shape_0 = D.shape[0]
        shape_1 = D.shape[1]
        if D.dtype == 'complex64':
            print_j_len = 2
        else:
            print_j_len = 4
        print_j_len_range = int(shape_1/print_j_len) #+ int(shape_1%print_j_len)
        for i in range(int(shape_0)):
            for j in range(int(print_j_len_range)):
                print (D[i:i+1, int(print_j_len*j):int(print_j_len*(j+1))])
                if j==int(print_j_len_range-1) and int(shape_1%print_j_len) != 0:
                    print (D[i:i+1, print_j_len*(j+1):])
        '''
        
        # Create Mel filterbank matrix
        mel_fb = librosa.filters.mel(sr=uni_sample,
                                     n_fft=1024,
                                     n_mels=64)
        '''
        print ('py extract, librosa.filters.mel.dtype=', mel_fb.dtype)
        print ('shape=', mel_fb.shape)
        print ('output mel_fb=')
        shape_0 = mel_fb.shape[0]
        shape_1 = mel_fb.shape[1]
        if mel_fb.dtype == 'complex64':
            print_j_len = 2
        else:
            print_j_len = 4
        print_j_len_range = int(shape_1/print_j_len) #+ int(shape_1%print_j_len)
        for i in range(int(shape_0)):
            for j in range(int(print_j_len_range)):
                print (mel_fb[i:i+1, int(print_j_len*j):int(print_j_len*(j+1))])
                if j==int(print_j_len_range-1) and int(shape_1%print_j_len) != 0:
                    print (mel_fb[i:i+1, print_j_len*(j+1):])
        '''
        
        # Transform to Mel frequency scale
        S = np.dot(mel_fb, np.abs(D) ** 2).T
        '''
        print ('py extract, S.dtype=', S.dtype)
        print ('shape=', S.shape)
        print ('output S=')
        shape_0 = S.shape[0]
        shape_1 = S.shape[1]
        if S.dtype == 'complex64':
            print_j_len = 2
        else:
            print_j_len = 4
        print_j_len_range = int(shape_1/print_j_len) #+ int(shape_1%print_j_len)
        for i in range(int(shape_0)):
            for j in range(int(print_j_len_range)):
                print (S[i:i+1, int(print_j_len*j):int(print_j_len*(j+1))])
                if j==int(print_j_len_range-1) and int(shape_1%print_j_len) != 0:
                    print (S[i:i+1, print_j_len*(j+1):])
        '''
        
        # Apply log nonlinearity and return as float32
        return librosa.power_to_db(S, ref=np.max, top_db=None)

    def pad_truncate(x, length, pad_value=0):
        """Pad or truncate an array to a specified length.

        Args:
            x (array_like): Input array.
            length (int): Target length.
            pad_value (number): Padding value.

        Returns:
            array_like: The array padded/truncated to the specified length.
        """
        x_len = len(x)
        if x_len > length:
            x = x[:length]
        elif x_len < length:
            padding = np.full((length - x_len,) + x.shape[1:], pad_value)
            x = np.concatenate((x, padding))

        return x

    # for original record
    # inputs = np.sum(inputs, axis = 1)
    # inputs = inputs/2**15

    # silence_thrshold
    #top_index = np.argpartition(abs(inputs), -int(sample_rate*0.3))[-int(sample_rate*0.3):]
    #top_max = abs(inputs)[top_index]

    #if 20 * np.log10(np.mean(top_max)) < -32:
    #    return (99, 'too_small')
        # return(99, 1)
    '''
    # cut_silence
    y=inputs
    top_db=48
    ref=np.max
    frame_length=2048
    hop_length=512
    
    ### test def _signal_to_frame_nonsilent(y, frame_length=2048, hop_length=512, top_db=60, ref=np.max):
    y_mono = librosa.core.to_mono(y)
    print ('py y_mono.dtype=', y_mono.dtype)
    print ('y_mono.shape=', y_mono.shape)
    
    print ('y_mono=')
    #shape_0 = inputs.shape[0]
    shape_1 = y_mono.shape[0]
    if y_mono.dtype == 'complex64':
        print_j_len = 2
    else:
        print_j_len = 4
    print_j_len_range = int(shape_1/print_j_len) #+ int(shape_1%print_j_len)
    for j in range(print_j_len_range):
        print(y_mono[print_j_len*j:print_j_len*(j+1)])
        if j==print_j_len_range-1 and int(shape_1%print_j_len) != 0:
            print(y_mono[print_j_len*(j+1):])
    
    ## test feature.rmse
    pad_mode='reflect'
    y_pad = np.pad(y_mono, int(frame_length // 2), mode=pad_mode)
    print ('py np.pad(y_mono, int(frame_length // 2), mode=pad_mode).dtype=', y_pad.dtype)
    print ('np.pad(y_mono, int(frame_length // 2), mode=pad_mode).shape=', y_pad.shape)
    print ('np.pad(y_mono, int(frame_length // 2), mode=pad_mode)=')
    #shape_0 = inputs.shape[0]
    shape_1 = y_pad.shape[0]
    if y_pad.dtype == 'complex64':
        print_j_len = 2
    else:
        print_j_len = 4
    print_j_len_range = int(shape_1/print_j_len) #+ int(shape_1%print_j_len)
    for j in range(print_j_len_range):
        print(y_pad[print_j_len*j:print_j_len*(j+1)])
        if j==print_j_len_range-1 and int(shape_1%print_j_len) != 0:
            print(y_pad[print_j_len*(j+1):])
    
    x = librosa.util.frame(y_pad,
                       frame_length=frame_length,
                       hop_length=hop_length)
    print ('py librosa.util.frame.dtype=', x.dtype)
    print ('librosa.util.frame.shape=', x.shape)
    print ('librosa.util.frame=')
    shape_0 = x.shape[0]
    shape_1 = x.shape[1]
    if x.dtype == 'complex64':
        print_j_len = 2
    else:
        print_j_len = 4
    print_j_len_range = int(shape_1/print_j_len) #+ int(shape_1%print_j_len)
    for i in range(int(shape_0)):
        for j in range(int(print_j_len_range)):
            print (x[i:i+1, int(print_j_len*j):int(print_j_len*(j+1))])
            if j==int(print_j_len_range-1) and int(shape_1%print_j_len) != 0:
                print (x[i:i+1, print_j_len*(j+1):])
    
    power = np.sqrt(np.mean(np.abs(x)**2, axis=0, keepdims=True))
    print ('py power.dtype=', power.dtype)
    print ('power.shape=', power.shape)
    print ('power=')
    #shape_0 = inputs.shape[0]
    shape_1 = power.shape[1]
    if power.dtype == 'complex64':
        print_j_len = 2
    else:
        print_j_len = 4
    print_j_len_range = int(shape_1/print_j_len) #+ int(shape_1%print_j_len)
    for j in range(print_j_len_range):
        print(power[print_j_len*j:print_j_len*(j+1)])
        if j==print_j_len_range-1 and int(shape_1%print_j_len) != 0:
            print(power[print_j_len*(j+1):])
    
    mse_alt0 = power**2
    print ('py mse_alt0.dtype=', mse_alt0.dtype)
    print ('mse_alt0.shape=', mse_alt0.shape)
    print ('mse_alt0=')
    #shape_0 = inputs.shape[0]
    shape_1 = mse_alt0.shape[1]
    if mse_alt0.dtype == 'complex64':
        print_j_len = 2
    else:
        print_j_len = 4
    print_j_len_range = int(shape_1/print_j_len) #+ int(shape_1%print_j_len)
    for j in range(print_j_len_range):
        print(mse_alt0[print_j_len*j:print_j_len*(j+1)])
        if j==print_j_len_range-1 and int(shape_1%print_j_len) != 0:
            print(mse_alt0[print_j_len*(j+1):])
    
    mse_alt1 = np.mean(np.abs(x)**2, axis=0, keepdims=True)
    print ('py mse_alt1.dtype=', mse_alt1.dtype)
    print ('mse_alt1.shape=', mse_alt1.shape)
    print ('mse_alt1=')
    #shape_0 = inputs.shape[0]
    shape_1 = mse_alt1.shape[1]
    if mse_alt1.dtype == 'complex64':
        print_j_len = 2
    else:
        print_j_len = 4
    print_j_len_range = int(shape_1/print_j_len) #+ int(shape_1%print_j_len)
    for j in range(print_j_len_range):
        print(mse_alt1[print_j_len*j:print_j_len*(j+1)])
        if j==print_j_len_range-1 and int(shape_1%print_j_len) != 0:
            print(mse_alt1[print_j_len*(j+1):])
    
    mse = librosa.feature.rmse(y=y_mono,
                       frame_length=frame_length,
                       hop_length=hop_length)**2
    print ('py mse.dtype=', mse.dtype)
    print ('mse.shape=', mse.shape)
    print ('mse=')
    #shape_0 = inputs.shape[0]
    shape_1 = mse.shape[1]
    if mse.dtype == 'complex64':
        print_j_len = 2
    else:
        print_j_len = 4
    print_j_len_range = int(shape_1/print_j_len) #+ int(shape_1%print_j_len)
    for j in range(print_j_len_range):
        print(mse[print_j_len*j:print_j_len*(j+1)])
        if j==print_j_len_range-1 and int(shape_1%print_j_len) != 0:
            print(mse[print_j_len*(j+1):])
    
    tmp = librosa.core.power_to_db(mse.squeeze(),
                             ref=ref,
                             top_db=None)
    print ('py librosa.core.power_to_db.dtype=', tmp.dtype)
    print ('librosa.core.power_to_db.shape=', tmp.shape)
    print ('librosa.core.power_to_db=')
    #shape_0 = inputs.shape[0]
    shape_1 = tmp.shape[0]
    if tmp.dtype == 'complex64':
        print_j_len = 2
    else:
        print_j_len = 4
    print_j_len_range = int(shape_1/print_j_len) #+ int(shape_1%print_j_len)
    for j in range(print_j_len_range):
        print(tmp[print_j_len*j:print_j_len*(j+1)])
        if j==print_j_len_range-1 and int(shape_1%print_j_len) != 0:
            print(tmp[print_j_len*(j+1):])
    
    tmp = (tmp > - top_db)
    print ('py librosa.core.power_to_db > - top_db.dtype=', tmp.dtype)
    print ('librosa.core.power_to_db > - top_db.shape=', tmp.shape)
    print ('librosa.core.power_to_db > - top_db=')
    #shape_0 = inputs.shape[0]
    shape_1 = tmp.shape[0]
    if tmp.dtype == 'complex64':
        print_j_len = 2
    else:
        print_j_len = 4
    print_j_len_range = int(shape_1/print_j_len) #+ int(shape_1%print_j_len)
    for j in range(print_j_len_range):
        print(tmp[print_j_len*j:print_j_len*(j+1)])
        if j==print_j_len_range-1 and int(shape_1%print_j_len) != 0:
            print(tmp[print_j_len*(j+1):])
    
    non_silent = librosa.effects._signal_to_frame_nonsilent(inputs,top_db=48) 
    print ('py librosa.effects._signal_to_frame_nonsilent, non_silent.dtype=', non_silent.dtype)
    print ('librosa.effects._signal_to_frame_nonsilent: non_silent.shape=', non_silent.shape)
    print ('librosa.effects._signal_to_frame_nonsilent: non_silent=')
    #shape_0 = inputs.shape[0]
    shape_1 = non_silent.shape[0]
    if non_silent.dtype == 'complex64':
        print_j_len = 2
    else:
        print_j_len = 4
    print_j_len_range = int(shape_1/print_j_len) #+ int(shape_1%print_j_len)
    for j in range(print_j_len_range):
        print(non_silent[print_j_len*j:print_j_len*(j+1)])
        if j==print_j_len_range-1 and int(shape_1%print_j_len) != 0:
            print(non_silent[print_j_len*(j+1):])    
    
    nonzero = np.flatnonzero(non_silent)

    if nonzero.size > 0:
        # Compute the start and end positions
        # End position goes one frame past the last non-zero
        start = int(librosa.core.frames_to_samples(nonzero[0], hop_length))
        end = min(inputs.shape[-1],
                  int(librosa.core.frames_to_samples(nonzero[-1] + 1, hop_length)))
    
    print ('py nonzero.dtype=', nonzero.dtype)
    print ('nonzero.shape=', nonzero.shape)
    print ('nonzero.size=', nonzero.size)
    print ('nonzero[0]=', nonzero[0])
    print ('nonzero[-1]=', nonzero[-1])
    print ('inputs.shape[0]=', inputs.shape[0])
    print ('inputs.shape[-1]=', inputs.shape[-1])
    print ('librosa.core.frames_to_samples(nonzero[0], hop_length)=', librosa.core.frames_to_samples(nonzero[0], hop_length))
    print ('librosa.core.frames_to_samples(nonzero[-1] + 1, hop_length)=', librosa.core.frames_to_samples(nonzero[-1] + 1, hop_length))
    '''
    trim_out, duration = librosa.effects.trim(inputs, top_db=48)
    '''
    print ("py librosa.effects.trim, duration.dtype= ", duration.dtype)
    print ("librosa.effects.trim, duration= ", duration)
    print ("librosa.effects.trim, trim_out.dtype= ", trim_out.dtype)
    print ("librosa.effects.trim, trim_out.shape= ", trim_out.shape)
    print ("librosa.effects.trim, trim_out= ")
    for j in range(int(trim_out.shape[0]/4)):
        print (trim_out[int(4*j):int(4*(j+1))])
        if j==int(int(trim_out.shape[0]/4)-1) and int((trim_out.shape[0])%4) != 0:
            print (trim_out[4*(j+1):])
    '''
    
    # transfer input to mel
    mel = extract(trim_out, sample_rate)
    '''
    print ('py extract, mel.dtype=', mel.dtype)
    print ('shape=', mel.shape)
    print ('output mel=')
    shape_0 = mel.shape[0]
    shape_1 = mel.shape[1]
    if mel.dtype == 'complex64':
        print_j_len = 2
    else:
        print_j_len = 4
    print_j_len_range = int(shape_1/print_j_len) #+ int(shape_1%print_j_len)
    for i in range(int(shape_0)):
        for j in range(int(print_j_len_range)):
            print (mel[i:i+1, int(print_j_len*j):int(print_j_len*(j+1))])
            if j==int(print_j_len_range-1) and int(shape_1%print_j_len) != 0:
                print (mel[i:i+1, print_j_len*(j+1):])
    '''
    #mel = (mel + 150) / 150

    # pad to 128
    mel = pad_truncate(mel, 126)
    '''
    print ('py pad_truncate, mel.dtype=', mel.dtype)
    print ('shape=', mel.shape)
    print ('output mel=')
    shape_0 = mel.shape[0]
    shape_1 = mel.shape[1]
    if mel.dtype == 'complex64':
        print_j_len = 2
    else:
        print_j_len = 4
    print_j_len_range = int(shape_1/print_j_len) #+ int(shape_1%print_j_len)
    for i in range(int(shape_0)):
        for j in range(int(print_j_len_range)):
            print (mel[i:i+1, int(print_j_len*j):int(print_j_len*(j+1))])
            if j==int(print_j_len_range-1) and int(shape_1%print_j_len) != 0:
                print (mel[i:i+1, print_j_len*(j+1):])
    '''
    
    return mel

def detection_keras (model, mel):
    mel = (mel + 150) / 150
    '''
    print ('py (mel + 150) / 150, mel.dtype=', mel.dtype)
    print ('shape=', mel.shape)
    print ('output mel=')
    shape_0 = mel.shape[0]
    shape_1 = mel.shape[1]
    if mel.dtype == 'complex64':
        print_j_len = 2
    else:
        print_j_len = 4
    print_j_len_range = int(shape_1/print_j_len) #+ int(shape_1%print_j_len)
    for i in range(int(shape_0)):
        for j in range(int(print_j_len_range)):
            print (mel[i:i+1, int(print_j_len*j):int(print_j_len*(j+1))])
            if j==int(print_j_len_range-1) and int(shape_1%print_j_len) != 0:
                print (mel[i:i+1, print_j_len*(j+1):])
    '''
    # normalize the mel
    # mel = (mel.reshape((128,64,1))-mean_mel)/(mean_mel+random_noise)
    # 2-sec predict
    #print('mel.dtype=', mel.dtype)
    time_measure = time.time()
    pred = model.predict(mel.reshape((1, 126, 64, 1)))
    time_measure = time.time() - time_measure
    '''
    print('pred[0]=', pred[0])
    print('np.argmax(pred[0])=', np.argmax(pred[0]))
    print('pred[1]=', pred[1])
    print('np.argmax(pred[1])=', np.argmax(pred[1]))
    print('pred[2]=', pred[2])
    print('np.argmax(pred[2])=', np.argmax(pred[2]))
    print('pred[3]=', pred[3])
    print('np.argmax(pred[3])=', np.argmax(pred[3]))
    '''
    pred = (pred[0] + pred[1] + pred[2] + pred[3]) / 4
    '''
    print('pred=', pred)
    print('np.argmax(pred)=', np.argmax(pred))
    print('pred[0][np.argmax(pred)]=', pred[0][np.argmax(pred)])
    '''
    # print(np.argmax(pred, axis = 1), pred[0,np.argmax(pred, axis = 1)])

    # return(np.argmax(pred, axis = 1)[0], pred[0,np.argmax(pred, axis = 1)])

    if np.max(pred) > 0.1:
        return (np.argmax(pred), pred, time_measure)
    else:
        return (99, 1, time_measure)

    return (np.argmax(pred), pred, time_measure)


def detection_tflite (select_dtype, interpreter, input_details, output_details, mel):
    # 2-sec predict    
    if select_dtype == 0: # int8     
        mel = (mel + 150) / 150 * 255 - 128
        #print('mel=', mel)
        mel = mel.astype(np.int8)
    elif select_dtype == 1: # uint8    
        mel = (mel + 150) / 150 * 255
        mel = mel.astype(np.uint8)
    else: # float32        
        mel = (mel + 150) / 150
        #print('mel=', mel)
        if select_dtype == 2: # fp16
            #mel = mel.astype(np.float16)
            mel = mel.astype(np.float32)
        else:
            mel = mel.astype(np.float32)
        
    mel = mel.reshape((1, 126, 64, 1))
        
    interpreter.set_tensor(input_details[0]['index'], mel)
    time_measure = time.time()
    interpreter.invoke()
    time_measure = time.time() - time_measure
    pred = interpreter.get_tensor(output_details[0]["index"])
    pred = np.append( pred, interpreter.get_tensor(output_details[1]["index"]), axis = 0)
    pred = np.append( pred, interpreter.get_tensor(output_details[2]["index"]), axis = 0)
    pred = np.append( pred, interpreter.get_tensor(output_details[3]["index"]), axis = 0)
    #print('pred.dtype=', pred.dtype)
    pred = pred.astype(np.float32)
    '''  
    #print('mel=', mel)
    print('pred[0]=', pred[0])
    print('np.argmax(pred)=', np.argmax(pred[0]))
    print('pred[1]=', pred[1])
    print('np.argmax(pred[1])=', np.argmax(pred[1]))
    print('pred[2]=', pred[2])
    print('np.argmax(pred[2])=', np.argmax(pred[2]))
    print('pred[3]=', pred[3])
    print('np.argmax(pred[3])=', np.argmax(pred[3]))
    print('pred.dtype=', pred.dtype)
    print('pred.shape=', pred.shape)
    '''
    pred = (pred[0] + pred[1] + pred[2] + pred[3]) / 4
    '''
    print('pred=', pred)
    print('np.argmax(pred)=', np.argmax(pred))
    print('pred[np.argmax(pred)]=', pred[np.argmax(pred)])
    '''
    if select_dtype == 0: # int8
        if np.max(pred) > 0.1 * 255 - 128:
            return (np.argmax(pred), pred, time_measure)
        else:
            return (99, 1, time_measure)
    elif select_dtype == 1: # uint8 
        if np.max(pred) > 0.1 * 255:
            return (np.argmax(pred), pred, time_measure)
        else:
            return (99, 1, time_measure)
    else: # float32
        if np.max(pred) > 0.1:
            return (np.argmax(pred), pred, time_measure)
        else:
            return (99, 1, time_measure)
    
    return (np.argmax(pred), pred, time_measure)


def initial (target_device, select_dtype, select_dev, select_opt, sel_pred_threads):
    # https://coral.ai/docs/edgetpu/tflite-python/#load-tensorflow-lite-and-run-an-inference
    # https://ithelp.ithome.com.tw/articles/10225953?sc=hot
    # https://github.com/mattroos/EdgeTpuTesting/blob/master/test_edgetpu.ipynb
    # https://colab.research.google.com/github/google-coral/tutorials/blob/master/retrain_classification_ptq_tf2.ipynb#scrollTo=iBs0O7q_wlCN
    # https://gist.github.com/iwatake2222/e4c48567b1013cf31de1cea36c4c061c
    # https://github.com/google-coral/edgetpu/tree/master/benchmarks
       
    if select_dev == 0 or select_dev == 1: # tpu
        if select_dtype == 0: # int8
            if select_opt == 0: # non opt model
                tflite_model_path='../freesound_umedia_20_classes_skip_attention_model_0420_int8_edgetpu.tflite'
            else: # opt model
                tflite_model_path='../freesound_umedia_20_classes_skip_attention_model_0420_opt_int8_edgetpu.tflite'
        elif select_dtype == 1: # uint8
            if select_opt == 0: # non opt model
                tflite_model_path='../freesound_umedia_20_classes_skip_attention_model_0420_uint8_edgetpu.tflite'
            else: # opt model
                tflite_model_path='../freesound_umedia_20_classes_skip_attention_model_0420_opt_uint8_edgetpu.tflite'
    elif select_dev == 2: # cpu
        if select_dtype == 0: # int8
            if select_opt == 0: # non opt model
                tflite_model_path='../freesound_umedia_20_classes_skip_attention_model_0420_int8.tflite'
            else: # opt model
                tflite_model_path='../freesound_umedia_20_classes_skip_attention_model_0420_opt_int8.tflite'
        elif select_dtype == 1: # uint8
            if select_opt == 0: # non opt model
                tflite_model_path='../freesound_umedia_20_classes_skip_attention_model_0420_uint8.tflite'
            else: # opt model
                tflite_model_path='../freesound_umedia_20_classes_skip_attention_model_0420_opt_uint8.tflite'
        elif select_dtype == 2: # fp16
            if select_opt == 0: # non opt model
                tflite_model_path='../freesound_umedia_20_classes_skip_attention_model_0420_fp16.tflite'
            else: # opt model
                tflite_model_path='../freesound_umedia_20_classes_skip_attention_model_0420_opt_fp16.tflite'
        elif select_dtype == 3: # fp32fallback
            if select_opt == 0: # non opt model
                tflite_model_path='../freesound_umedia_20_classes_skip_attention_model_0420_fp32fallback.tflite'
            else: # opt model
                tflite_model_path='../freesound_umedia_20_classes_skip_attention_model_0420_opt_fp32fallback.tflite'
        else: # float32
            if select_opt == 0: # non opt model
                tflite_model_path='../freesound_umedia_20_classes_skip_attention_model_0420.tflite'
            else: # opt model
                tflite_model_path='../freesound_umedia_20_classes_skip_attention_model_0420_opt.tflite'
    
    if select_dev == 0 or select_dev == 1: # tpu using interpreter from tflite_runtime package
        import tensorflow as tf
        from tflite_runtime.interpreter import load_delegate
        from tflite_runtime.interpreter import Interpreter
        '''
        import edgetpu
        import edgetpu.basic.edgetpu_utils        
        devices = edgetpu.basic.edgetpu_utils.ListEdgeTpuPaths(edgetpu.basic.edgetpu_utils.EDGE_TPU_STATE_NONE)
        #print('len(devices)= ', len(devices))
        #print('edgetpu devices= ', devices)
        # Set identity of edge device to use, if any
        if len(devices) > 0:
            # Use the first device in the list
            if devices[0].startswith('/dev/apex'):
                target_device = 'pci'
            else:
                # Assuming device is on USB bus
                target_device = 'usb'
        '''
        time_measure = time.time()
        interpreter = Interpreter(model_path=tflite_model_path,
                                  model_content=None,
                                  experimental_delegates=[load_delegate('libedgetpu.so.1.0',
                                                                        {'device': target_device})])
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        time_measure = time.time() - time_measure
        return (interpreter, input_details, output_details, time_measure)
       
    elif select_dev == 1: # tpu using interpreter from full TensorFlow package
        import tensorflow as tf
        from tensorflow.lite.python.interpreter import load_delegate
        '''
        import edgetpu
        import edgetpu.basic.edgetpu_utils        
        devices = edgetpu.basic.edgetpu_utils.ListEdgeTpuPaths(edgetpu.basic.edgetpu_utils.EDGE_TPU_STATE_NONE)
        #print('edgetpu devices= ', devices)
        # Set identity of edge device to use, if any
        if len(devices) > 0:
            # Use the first device in the list
            if devices[0].startswith('/dev/apex'):
                target_device = 'pci'
            else:
                # Assuming device is on USB bus
                target_device = 'usb'
        '''
        time_measure = time.time()
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path,
                                          experimental_delegates=[tflite.load_delegate('libedgetpu.so.1.0',
                                                                                {'device': target_device})])
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        time_measure = time.time() - time_measure
        return (interpreter, input_details, output_details, time_measure)
        
    elif select_dev == 2: # cpu using interpreter from full TensorFlow package        
        import tensorflow as tf
        time_measure = time.time()
        #interpreter = tf.lite.Interpreter(model_path=tflite_model_path \
        #                                  ,num_threads=sel_pred_threads)
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        time_measure = time.time() - time_measure
        return (interpreter, input_details, output_details, time_measure)
    
    else: # cpu using keras
        # https://blog.victormeunier.com/posts/keras_multithread/
        '''
        import sys
        stdout = sys.stdout
        sys.stdout = open('/dev/null', 'w')       
        import tensorflow as tf
        import keras
        '''
        from keras.models import Sequential,Model, load_model
        '''
        sys.stdout = stdout
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import logging
        logger = tf.get_logger()
        logger.setLevel(logging.ERROR)
        '''
        time_measure = time.time()
        model = load_model("../freesound_umedia_20_classes_skip_attention_model_0420.h5")
        time_measure = time.time() - time_measure
        return (model, time_measure)


def run_single_dev ( target_device \
                    ,time_load_model ,time_1st_pred ,time_pred ,time_total ,accuracy_cnt \
                    ,dir_wav ,total_filecount \
                    ,select_dev ,select_dtype ,select_opt ,sel_pred_threads \
                    ,show_title):
    
    load_model = 1
    time_load_model[select_dev, select_dtype, sel_pred_threads] = 0
    time_1st_pred[select_dev, select_dtype, sel_pred_threads] = 0
    time_pred[select_dev, select_dtype, sel_pred_threads] = 0
    time_total[select_dev, select_dtype, sel_pred_threads] = 0
    accuracy_cnt[select_dev, select_dtype, sel_pred_threads] = 0
    
    for i in range(0, total_filecount):
    #for i in range(7, 8):
        if i == 0:
            #file = "MS200199500008.0.1597747308.wav"
            file = dir_wav + "alarm1.wav"
        elif i == 1:
            file = dir_wav + "bark1.wav"
        elif i == 2:
            file = dir_wav + "boiling1.wav"
        elif i == 3:
            file = dir_wav + "cough1.wav"
        elif i == 4:
            file = dir_wav + "cry2.wav"
        elif i == 5:
            file = dir_wav + "doorbell2.wav"
        elif i == 6:
            file = dir_wav + "fall1.wav"
        elif i == 7:
            file = dir_wav + "groan1.wav"
        elif i == 8:
            file = dir_wav + "screaming1.wav"
        elif i == 9:
            file = dir_wav + "shatter1.wav"
        elif i == 10:
            file = dir_wav + "sneeze1.wav"
        elif i == 11:
            file = dir_wav + "snoring2.wav"
        elif i == 12:
            file = dir_wav + "speech1.wav"
        elif i == 13:
            file = dir_wav + "telephone2.wav"
        elif i == 14:
            file = dir_wav + "toilet1.wav"
        elif i == 15:
            file = dir_wav + "walk1.wav"
        else : #i == 16:
            file = dir_wav + "water1.wav"

        #print('')
        #print('i=', i)
        x,sr = librosa.load(file, sr=None, dtype=np.float64)    
        xf = np.zeros(63488)
        if len(x) < 63488:
            xf [0:len(x)] = x
        else:
            xf [0:63488] = x [0:63488]
        mel_o = detection_input (xf, sr, 32000)
        '''
        pred = detection_keras(model, mel_o)
        #pred = detection(model, x, sr, 44100)
        #print('type(pred)=', type(pred))
        #print('len(pred)=', len(pred))
        print(class_list[pred[0]])
        '''
        if (select_dev == 3): # cpu using keras
            if (load_model == 1):
                model ,time_load_model[ select_dev ,select_dtype ,sel_pred_threads] \
                = initial (  target_device \
                            ,select_dtype \
                            ,select_dev \
                            ,select_opt \
                            ,sel_pred_threads)
            time_measure = time.time()
            pred = detection_keras (model, mel_o)
        else: # cpu or tpu using tflite
            if (load_model == 1):
                interpreter \
               ,input_details \
               ,output_details \
               ,time_load_model[select_dev ,select_dtype ,sel_pred_threads] \
               = initial (   target_device \
                            ,select_dtype \
                            ,select_dev \
                            ,select_opt \
                            ,sel_pred_threads)
            time_measure = time.time()
            pred = detection_tflite (select_dtype, interpreter, input_details, output_details, mel_o)
    
        if (load_model == 1):
            time_total[select_dev ,select_dtype ,sel_pred_threads] = time.time() - time_measure
            time_pred[select_dev ,select_dtype ,sel_pred_threads] = pred[2]
            time_1st_pred[select_dev ,select_dtype ,sel_pred_threads] = pred[2]
            load_model = 0
        else:
            time_total[select_dev ,select_dtype ,sel_pred_threads] += time.time() - time_measure
            time_pred[select_dev ,select_dtype ,sel_pred_threads] += pred[2]
        '''
        print('time_1st_pred[select_dev ,select_dtype, sel_pred_threads]=', time_1st_pred[select_dev ,select_dtype ,sel_pred_threads])
        print('time_pred[select_dev ,select_dtype ,sel_pred_threads]=', time_pred[select_dev ,select_dtype ,sel_pred_threads])
        print('time_total[select_dev ,select_dtype ,sel_pred_threads]=', time_total[select_dev ,select_dtype ,sel_pred_threads])
        '''
        file = ((file.replace(dir_wav, '')).rstrip('.wav')).rstrip('12')
        if file == class_list_file[pred[0]].lower():
            accuracy_cnt[select_dev ,select_dtype ,sel_pred_threads] += 1
        '''
        print('file=', file)
        print(class_list_file[pred[0]].lower())
        print('file=', file)
        print(class_list_file[pred[0]].lower())
        print(accuracy_cnt[select_dev ,select_dtype ,sel_pred_threads])
        '''
    # end loop, print result
    result_print = []
    if select_dev == 0:
        result_print = '  TPU, runtime |'
    elif select_dev == 1:
        result_print = '  TPU, tflite  |'
    elif select_dev == 2:
        result_print = '  CPU, tflite  |'
    else:
        result_print = '  CPU, Keras   |'
    
    if select_opt == 0:
        result_print += '     N     |'
    else:
        result_print += '     Y     |'

    if select_dtype == 0:
        result_print += '    int8   |'
    elif select_dtype == 1:
        result_print += '   uint8   |'
    elif select_dtype == 2:
        result_print += '     fp16  |'
    elif select_dtype == 3:
        result_print += '  fp32_fb  |'
    elif select_dtype == 4:
        result_print += '     fp32  |'
    else:
        result_print += '     fp64  |'

    result_print += '      ' + str(sel_pred_threads) + '       |'
    result_print += '    ' + str(round(accuracy_cnt[select_dev ,select_dtype ,sel_pred_threads]/total_filecount*100, 2)) + '    |'
    
    if len(str(round(time_load_model[select_dev ,select_dtype ,sel_pred_threads], 5))) == 5:
        result_print += '      ' + str(round(time_load_model[select_dev ,select_dtype ,sel_pred_threads], 5)) + '    |'
    elif len(str(round(time_load_model[select_dev ,select_dtype ,sel_pred_threads], 5))) == 6:
        result_print += '     ' + str(round(time_load_model[select_dev ,select_dtype ,sel_pred_threads], 5)) + '    |'
    elif len(str(round(time_load_model[select_dev ,select_dtype ,sel_pred_threads], 5))) == 7:
        result_print += '    ' + str(round(time_load_model[select_dev ,select_dtype ,sel_pred_threads], 5)) + '    |'
    else:
        result_print += '   ' + str(round(time_load_model[select_dev ,select_dtype ,sel_pred_threads], 5)) + '    |'
        
    if len(str(round(time_1st_pred[select_dev ,select_dtype ,sel_pred_threads], 5))) == 4:
        result_print += '    ' + str(round(time_1st_pred[select_dev ,select_dtype ,sel_pred_threads], 5)) + '      |'
    elif len(str(round(time_1st_pred[select_dev ,select_dtype ,sel_pred_threads], 5))) == 5:
        result_print += '    ' + str(round(time_1st_pred[select_dev ,select_dtype ,sel_pred_threads], 5)) + '     |'
    elif len(str(round(time_1st_pred[select_dev ,select_dtype ,sel_pred_threads], 5))) == 6:
        result_print += '    ' + str(round(time_1st_pred[select_dev ,select_dtype ,sel_pred_threads], 5)) + '    |'
    else:
        result_print += '    ' + str(round(time_1st_pred[select_dev ,select_dtype ,sel_pred_threads], 5)) + '   |'
    
    if len(str(round(time_pred[select_dev ,select_dtype ,sel_pred_threads]/total_filecount, 5))) == 4:
        result_print += '     ' + str(round(time_pred[select_dev ,select_dtype ,sel_pred_threads]/total_filecount, 5)) + '       |'
    elif len(str(round(time_pred[select_dev ,select_dtype ,sel_pred_threads]/total_filecount, 5))) == 5:
        result_print += '     ' + str(round(time_pred[select_dev ,select_dtype ,sel_pred_threads]/total_filecount, 5)) + '      |'
    elif len(str(round(time_pred[select_dev ,select_dtype ,sel_pred_threads]/total_filecount, 5))) == 6:
        result_print += '     ' + str(round(time_pred[select_dev ,select_dtype ,sel_pred_threads]/total_filecount, 5)) + '     |'
    else:
        result_print += '     ' + str(round(time_pred[select_dev ,select_dtype ,sel_pred_threads]/total_filecount, 5)) + '    |'
    
    if round(time_total[select_dev ,select_dtype ,sel_pred_threads]/total_filecount, 5) >= 10:
        result_print += '      ' + str(round(time_total[select_dev ,select_dtype ,sel_pred_threads]/total_filecount, 5))
    else:
        result_print += '       ' + str(round(time_total[select_dev ,select_dtype ,sel_pred_threads]/total_filecount, 5))
    
    if show_title == 1:
        print('')
        print('  Benchmark result:')
        print('     Device    | Opt model | Data type | Pred_threads | Accuracy(%) | Load model(s) | 1st pred(s)  | Avg 17 pred(s) | Avg 17 set_io + pred(s)')
    
    print(result_print)


###################### main ######################    
import edgetpu
import edgetpu.basic.edgetpu_utils
tpu_devices = edgetpu.basic.edgetpu_utils.ListEdgeTpuPaths(edgetpu.basic.edgetpu_utils.EDGE_TPU_STATE_NONE)
#print('len(devices)= ', len(devices))
#print('edgetpu devices= ', devices)
# Set identity of edge device to use, if any
if len(tpu_devices) > 0:
    run_tpu_runtime = 1
    # Use the first device in the list
    if tpu_devices[0].startswith('/dev/apex'):
        target_device = 'pci'
    else:
        # Assuming device is on USB bus
        target_device = 'usb'
    print('Note: Detect edgetpu devices is '+ target_device + '. Run TPU & CPU.')
else:
    run_tpu_runtime = 0
    target_device = 'none'
    print('Warning: Detect edgetpu devices is '+ target_device + '. Run without TPU.')

run_cpu_tf = 1
run_keras = 0
#run_tpu_tf = 1

total_filecount = 17
dir_wav = "../../Data_for_demo/32k_mono_32bit/"

select_dev = 3
select_dtype = 5
sel_pred_threads = 4
time_load_model = np.zeros((select_dev+1, select_dtype+1, sel_pred_threads))
time_1st_pred = np.zeros((select_dev+1, select_dtype+1, sel_pred_threads))
time_pred = np.zeros((select_dev+1, select_dtype+1, sel_pred_threads))
time_total = np.zeros((select_dev+1, select_dtype+1, sel_pred_threads))
accuracy_cnt = np.zeros((select_dev+1, select_dtype+1, sel_pred_threads))

import warnings
warnings.filterwarnings('ignore')
show_title = 1

if run_tpu_runtime == 1:
    #for i in range(0, 2):
    for i in range(0, 2):
        for j in range(0, 2):
            run_single_dev ( target_device \
                            ,time_load_model ,time_1st_pred ,time_pred ,time_total ,accuracy_cnt \
                            ,dir_wav ,total_filecount \
                            ,select_dev=0 ,select_dtype=i ,select_opt=j ,sel_pred_threads=1 \
                            ,show_title=show_title)
            show_title = 0

if run_cpu_tf == 1:
    for i in range(0, 5):
    #for i in range(0, 2):
        for j in range(0, 2):
            for k in range(1, 2): #5, 3):
                run_single_dev ( target_device \
                                ,time_load_model ,time_1st_pred ,time_pred ,time_total ,accuracy_cnt \
                                ,dir_wav ,total_filecount \
                                ,select_dev=2 ,select_dtype=i ,select_opt=j ,sel_pred_threads=k \
                                ,show_title=show_title)
                show_title = 0

if run_keras == 1:
    run_single_dev ( target_device \
                    ,time_load_model ,time_1st_pred ,time_pred ,time_total ,accuracy_cnt \
                    ,dir_wav ,total_filecount \
                    ,select_dev=3 ,select_dtype=3 ,select_opt=0 ,sel_pred_threads=1 \
                    ,show_title=show_title)
    show_title = 0

'''
if run_tpu_tf == 1:
    for i in range(0, 2):
        for j in range(0, 2):
            run_single_dev ( target_device \
                            ,time_load_model ,time_1st_pred ,time_pred ,time_total ,accuracy_cnt \
                            ,dir_wav ,total_filecount \
                            ,select_dev=1 ,select_dtype=i ,select_opt=j ,sel_pred_threads=1 \
                            ,show_title=show_title)
            show_title = 0
'''
   
    