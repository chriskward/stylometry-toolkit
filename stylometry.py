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

 		self._index , self._count = self._ngram_count()
		self._freq = self._count / np.sum(self_count)

 		self.ngrams_count = pd.Series(index = self._index, data = self._count)
        self.ngrams_freq = pd.Series(index = self._index, data = self._freq)


 	def _get_stride(self):
 		return self._n

 	# n is a managed attribute. _set_stride() will be called if the value of n is changed

 	n = property(fset = _set_stride, fget = _get_stride)


 	def _ngram_count(self):

 		# flatten() returns a copy of self.ngrams with the 2nd dimension removed
 		# we're going to successively shrink x as we count ngrams and need to leave
 		# self.ngrams unaltered

 		x = self.ngrams.flatten()

 		index , counts = list() , list()

 		while len(x) > 0:
 			bool_mask = x==x[0]					# find all occurances of the first n-gram in the sequence of n-grams
 			index.append(x[0])					# append this n-gram to the index
 			counts.append( np.sum(bool_mask) )	# append the count
 			x = x[~bool_mask]					# drop all occurances of this n-gram from the sequence and repeat till seq. is empty

 		return np.array(index) , np.array(counts)


 	def __getitem__ (self, index = None):

 		# permit the instances of the class to be indexed
 		# indexing will return elements of the sequence of n-grams
 		# displayed in text form rather than unicode (IS THIS THE MOST APPROPRIATE DATA TO RETURN?)

 		if index == None: return self.ngrams
 		else : return self.ngrams[index]


 	def __iter__(self):

 		# the iterator will also return text rather than unicode

 		self._index = 0
 		return self


 	def __next__(self):
 		if self._index < len(self.ngrams):
 			x = self.ngrams[self._index]
 			self._index += 1
 			return x

 		else: raise StopIteration 


 	def __repr__ (self):
        x = self.ngrams_count
        y = self.ngrams_freq
        return f'Label = {self.label} \r\r ' + str(x) + '\r\r' + str(y)
    

    def __str__ (self):
        return self._original_text
    

    def __len__ (self):
        return len(self.ngrams)




class TextDataset():

	def __init__ (self, *args, **kwargs):

		if not all(isinstance(x, TextSample) for x in args): raise TypeError("Requires TextSample objects as input")
		self._dataset = args

		# An instance of the class has two attributes; X, Y 
		# These are pandas dataframes, X contains relative frequencies/counts of each ngram
		# and Y the text sample label. The i-the row of X and Y correspond to the i-th Text Sample
		# The specifics of these data frames depend on a range of attributes that can either be set
		# at instantiation or adjusted later by calling the TextDataset instance.

		# To avoid code repetition, __init__ calls __call__, passing through the key word arguments

		self.__call__(**kwargs)



	def __call__(self, **kwargs):

		default_kwargs = {'n':2, 'counts':False , 'common':False, 'most_frequent':None}
		self.mode = {**default_kwargs, **kwargs}			# if any of the keys are no present, values default_kwargs will not be overwritten

		# set n for each TextSample() in the dataset
		for x in self._dataset: x.n = self.mode['n']

		# common = False  - we are interested in all ngrams that appear in all TextSamples() e.g. some ngrams may appear only once or only in a single TextSample()
		# this is the UNION of the sets of ngrams of each TextSample()

		if self.mode['common'] == False: global_index = reduce( np.union1d, [x.index for x in self._dataset] )

		# common = True  - we are only interested in the subset of ngrams that appear at least once in each TextSample() 
        # this is the INTERSECTION of the sets of ngrams of each TextSample()

        else: global_index = reduce( np.intersect1d, [x.index for x in self._dataset] )

        # do we want X to show counts or relative frequences

        if self.mode['counts'] == True: attribute = 'ngrams_count'
        else: attribute = 'ngrams_freq'

        global_data , labels = list() , list()

        for x in self._dataset:
        	sample_data = list()
        	labels.append(x.label)
        	x = getattr(x,attribute).reindex(global_index, fill_value=0)
        	global_data.append(x)

        self.X = pd.concat(global_data, axis=1).T
        self.Y = pd.DataFrame( data = labels, columns = ['label'])

        if self.mode['most_frequent'] != None:
            occurances = self.X.sum(axis=0).sort_values(ascending=False)
            new_index = occurances.nlargest(self.mode['most_frequent']).index
            self.X = self.X.loc[:,new_index]