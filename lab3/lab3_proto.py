import numpy as np
from lab3_tools import *

def words2phones(wordList, pronDict, addSilence=True, addShortPause=True):
   """ word2phones: converts word level to phone level transcription adding silence

   e.g. ["z","4","3"] -> ['sil', 'z', 'iy', 'r', 'ow', 'sp', 'f', 'ao', 'r', 'sp', 'th', 'r', 'iy', 'sp', 'sil']

   ['sil', 'z', 'iy', 'r', 'ow', 'sp', 'f', 'ao', 'r', 'sp', 'th', 'r', 'iy', 'sp', 'sil']
    Args:
       wordList: list of word symbols
       pronDict: pronunciation dictionary. The keys correspond to words in wordList
       addSilence: if True, add initial and final silence
       addShortPause: if True, add short pause model "sp" at end of each word
    Output:
       list of phone symbols
   """
   
   if addSilence:
      output = ["sil"]
   else:
      output = []

   for word in wordList:
      #one transcribed word
      for w in pronDict[word]:
         output.append(w)
      if addShortPause:
         output.append('sp')
   
   if addSilence:
      output.append("sil")
   return output

def forcedAlignment(lmfcc, phoneHMMs, phoneTrans):
    """ forcedAlignmen: aligns a phonetic transcription at the state level

    Args:
       lmfcc: NxD array of MFCC feature vectors (N vectors of dimension D)
              computed the same way as for the training of phoneHMMs
       phoneHMMs: set of phonetic Gaussian HMM models
       phoneTrans: list of phonetic symbols to be aligned including initial and
                   final silence

    Returns:
       list of strings in the form phoneme_index specifying, for each time step
       the state from phoneHMMs corresponding to the viterbi path.
    """

