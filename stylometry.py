# Chris Ward Jan 2023


import numpy as np
import pandas as pd
from functools import reduce


class TextSample():

	def __init__(self,text,label=0,n=2):

		self.original_text = text
		self.label = label

		# cast the text string to a 0D ndarray, unicode stored as uint32 
		
		text = np.array(text, dtype=np.str_)

		# create a view of the data buffer of the 0D text array
		# each index of _buffer is the 4byte integer repr. - a unicode codepoint
 		
 		self._buffer = np.frombuffer( text , dtype=np.uint32 )

 		# the remaining attributes are functions of n (as in n-grams)
 		# these are computed by the private method _set_stride()

 		self._set_stride(n):


 	def _set_stride(self, n):

		self._n = n
		window_shape = (self._n,)

		#create a sliding window view of the data buffer of size n
 		
 		self.unicode = np.lib.stride_tricks.sliding_window_view( self._buffer,window_shape )

 		#create a text view of the sliding window
 		#not strictly necessary, but increases human-readability of output

 		self.ngrams = self.unicode.view('U' + str(self._n))

 		# _ngram_count() method returns a tuple (n_gram index list, count of ngrams)

 		self.ngrams_unique , self.ngrams_count = self._ngram_count()
 		self.ngrams_freq = self.ngrams_count / np.sum(self.ngrams_count)


 	def _get_stride(self):
 		return self._n


 	# n is a managed attribute. _set_stride() will be called if the value of n is changed

 	n = property(fset = _set_stride, fget = _get_stride)


 	def _ngram_count(self):
 		pass

 		