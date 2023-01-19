# Chris Ward Jan 2023


import numpy as np
import pandas as pd
from functools import reduce


class TextSample():

	def __init__(self,text,label=0,n=2):

		self.original_text = text
		self.label = label

		#cast the text string to a 0D ndarray, unicode stored as uint32 
		text = np.array(text, dtype=np.str_)

		#create a view of the buffer of text
		#each inxed of _buffer is the 4byte integer repr. of each character
 		self._buffer = np.frombuffer( text , dtype=np.uint32 )

 		self._set_stride(n)

 				# managed / computed attributes are all functions of n?




 # The class requires the following attributes: counts of each ngram, rel-freq of each ngram
 # array containing sequence of ngrams that constitute input text eg ('Hello') -> He,el,ll,lo

 # recompute all the above when n is assigned, reassigned?

 