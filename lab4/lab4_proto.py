import torchaudio
import torch
from torchaudio import transforms
import torchaudio.transforms as transforms
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence 
from pyctcdecode import build_ctcdecoder
# DT2119, Lab 4 End-to-end Speech Recognition

# Variables to be defined --------------------------------------

''' 
train-time audio transform object, that transforms waveform -> spectrogram, with augmentation
''' 
train_audio_transform = nn.Sequential(transforms.MelSpectrogram(n_mels=80),
                                      transforms.FrequencyMasking(freq_mask_param=15),
                                      transforms.TimeMasking(time_mask_param=35)
)

'''
test-time audio transform object, that transforms waveform -> spectrogram, without augmentation 
'''
test_audio_transform = transforms.MelSpectrogram(n_mels=80)

#labels =


# Functions to be implemented ----------------------------------

def intToStr(labels):
    '''
        convert list of integers to string
    Args: 
        labels: list of ints
    Returns:
        string with space-separated characters
    '''
    code_int_list = {}

    for idx, char in enumerate("' abcdefghijklmnopqrstuvwxyz",1):
        code_int_list[idx]=char 

    output = []
    for l in labels:
        if l in code_int_list: 
            output.append(code_int_list[l]) 
        else:
            output.append('?')  

    return ''.join(output)

def strToInt(text):
    '''
        convert string to list of integers
    Args:
        text: string
    Returns:
        list of ints
    '''
    code_char_list = {}

    for idx, char in enumerate("' abcdefghijklmnopqrstuvwxyz"):
        code_char_list[char]=idx 

    output = []
    for t in text:
        if t in code_char_list: 
            output.append(code_char_list[t]) 
        else:
            output.append('?')  

    return output




def dataProcessing(data, transform):
    '''
    process a batch of speech data
    arguments:
        data: list of tuples, representing one batch. Each tuple is of the form
            (waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id)
        transform: audio transform to apply to the waveform
    returns:
        a tuple of (spectrograms, labels, input_lengths, label_lengths) 
        -   spectrograms - tensor of shape B x C x T x M 
            where B=batch_size, C=channel, T=time_frames, M=mel_band.
            spectrograms are padded the longest length in the batch.
        -   labels - tensor of shape B x L where L is label_length. 
            labels are padded to the longest length in the batch. 
        -   input_lengths - list of half spectrogram lengths before padding
        -   label_lengths - list of label lengths before padding
    '''
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = [] 
    
    for waveform, _, utterannce, *_ in data: 
        spec = transform(waveform).squeeze(0).transpose(0,1)
        spectrograms.append(spec)
        label = torch.tensor(strToInt(utterannce.lower()), dtype=torch.int)
        labels.append(label)
        input_lengths.append(spec.shape[0] //2)
        label_lengths.append(len(label))

    spectrograms = pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2,3)
    labels = pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths 


"""
def greedyDecoder(output, blank_label=28):
    '''
    decode a batch of utterances 
    arguments:
        output: network output tensor, shape B x T x C where B=batch_size, T=time_steps, C=characters
        blank_label: id of the blank label token
    returns:
        list of decoded strings
    '''
    out_lst = []
    prev_label = None
    predicted_labels_batch = torch.argmax(output,dim=2)#.flatten()
    predicted_labels_batch = predicted_labels_batch.cpu()#.tolist()
    for predicted_labels_utterance in predicted_labels_batch:
        out = ""
        for predlabel in predicted_labels_utterance:
            if predlabel != blank_label and predlabel != prev_label:
                out += intToStr([predlabel])
                prev_label = predlabel
        out_lst.append(out)
    return out_lst
"""

def greedyDecoder(output, blank_label=28):
    '''
    decode a batch of utterances 
    arguments:
        output: network output tensor, shape B x T x C where B=batch_size, T=time_steps, C=characters
        blank_label: id of the blank label token
    returns:
        list of decoded strings
    '''
    decoded_str = []
    for batch in output:
        best_path = torch.argmax(batch, dim=1)
        prev_label = blank_label
        decoded = []
        for label in best_path:
            if label != blank_label and label != prev_label:
                decoded.append(label.item())
            prev_label = label
        decoded_str.append(intToStr(decoded))
    return decoded_str 


def levenshteinDistance(ref,hyp):
    '''
    calculate levenshtein distance (edit distance) between two sequences
    arguments:
        ref: reference sequence
        hyp: sequence to compare against the reference
    output:
        edit distance (int)
    '''
    ref = list(ref)
    hyp = list(hyp)

    len_ref , len_hyp = len(ref), len(hyp)

    if len_ref == 0:
        return(len_hyp)
    if len_hyp == 0:
        return(len_ref)
    
    matrix = [[0 for _ in range(len_hyp+1)] for _ in range(len_ref +1)]
    
    for i in range(len_ref + 1):
        matrix[i][0]=i
    for j in range(len_hyp + 1):
        matrix[0][j] = j 

    for i in range(1,len_ref + 1):
        for j in range(1,len_hyp + 1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1 
            matrix[i][j] = min(matrix[i-1][j] + 1, matrix[i][j-1]+ 1, matrix[i-1][j-1]+cost)

    return matrix[len_ref][len_hyp]