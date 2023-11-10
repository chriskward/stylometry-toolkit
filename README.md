## Stylometry Toolkit
### pip install git+https://github.com/chriskward/stylometry-toolkit
<br>

The package contains two classes, **TextSample** and **TextDataset** and is compatible with the marjority of scikit-learn
models. My approach and methodology was heavily influenced by the work of [Grieve (2007)](https://www.researchgate.net/publication/301404533_Not_All_Character_N-grams_Are_Created_Equal_A_Study_in_Authorship_Attribution),
[Stamatatos (2006,2013,2022)](https://scholar.google.com/citations?user=xie8sAEAAAAJ&hl=en) and [Sari et al (2018)](https://aclanthology.org/C18-1029/)

<br>


[Brief explanation and demonstration of **TextSample** and **TextDataset**](demo.ipynb)

[Authorship Attribution using the Victorian Authors Dataset from the UCI ML Repository](authorship-attribution.ipynb)



<br>

***
<br>

Stylometry involves the statistical analysis of text and writing style to illuminate characteristics of the writer.
Whilst the field borrows heavily from other areas of Natural Language Processing, there are a range of distinctions
that mean existing NLP tools are not always appropriate.

This repository grew from my masters thesis researching approaches to plagiarism detection;

    Given a collection of writers and a training set comprising 
    past examples of their writing, can we build a classification
    system that can correctly classify new text samples to the
    correct author.


The traditional and most widely studied approach to this sort of problem involves hyperplane classifiers and 
extensive feature engineering and is not too distinct from the bag-of-words approach to NLP. However, interest typically
centres not on words, but *character n-grams*.

More recent methods (that are in general far more effective) treat text as a sequence of symbols (or words/subword tokens)
and aim to model writing style as some complex autoregressive function over a such a sequence (succesful algorithms often
employ LSTMs or Transformer Encoders). This approach is discussed further in [Markov Lanugage Models](/markov-language-models)


<br>

****
<br>

This package focuses on the traditional character n-gram approach to stylometry and author identification. It includes 
a range of computationally efficient tools to process a collection of text samples and:

* express the text samples in character n-grams (for any n)
* dynamically change the value of n
* display relative frequencies or counts of n-grams
* select only a subset of all n-grams in the dataset

Instances of **TextDataset** expose the attributes X and Y. These are Pandas Dataframes displaying the data 
in chosen form (e.g. value of n, feature subset, counts/frequencies) and are compatible with all relevant sci-kit learn
models.

Both classes make use of np.frombuffer and np.lib.stride_tricks to directly access the nd.array data buffer and
dynamically adjust the stride and view. This is far more computationally efficient than the obvious ways to achieve this
functionality using for loops and iteration but does require numpy.__version\__ == 1.26 (see environment,yml) and is not
wholly supported by the numpy documentation so may not work with future versions.

<br>

****
<br>
