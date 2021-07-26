# Synonym Detection
This directory contains the first steps toward synonym detection for use in conjunction with iKnow/iFind, through the use of the word embedding models Word2Vec and fastText. This README contains information on the layout of the synonymdetection package, important notes before diving in, and a demo at the end!

## Relation to iKnow
The use of the synonymdetection package does not require [iKnow](https://github.com/intersystems/iknow) for basic functionality. If a user wishes to train a model using iKnow to tokenize [entities](https://github.com/intersystems/iknow/wiki/Entities), then it is necessary to also install the iknowpy package, available through PyPI:

    pip install iknowpy

If you've already got a model you want to use, or you don't want to tokenize entities for the sake of training, you can forego the installation of the iknowpy package and stick to just the synonymdetection package.

## File layout
The synonymdetection package is laid out as follows:

- corpora (dir): The directory to hold corpora for training models. If you use the iKnow entity preprocessing functionality in the module, it will automatically output the processed corpus here. You don't have to place a corpus in this directory to use it for training.
    - examplecorpus.txt: A small corpus used to train the superficial example model included in this repo for exploratory purposes.
- models (dir): 
    - word2vec (dir):
        - vectors (dir): The directory where Word2Vec vectors are automatically saved to and loaded from. Saving just vectors cuts down on storage size, memory, and time required for runtime processes
            - w2v_example: Vectors for the example model trained on examplecorpus.txt, saved in "word2vec format" to cut down on size.
        - trained_models (dir): The directory where trained Word2Vec models are optionally saved to and loaded from. If one wishes to update the model later, it must be saved here (by default, it will be saved), otherwise you can just save and use the vectors.
            - w2v_example: The actual model trained on examplecorpus.txt
    - fasttext (dir): A directory where trained fastText models are automatically saved to and loaded from. Each model is *automatically* saved in "facebook format" as a .bin file which encodes both the vectors and the model. Does not contain an example model as even superficial fastText models are too large.
- datasets (dir): The directory containing word pair files for use in modelevaluation.py. Also the directory where the output of modelevaluation.py will atuomatically be saved.
    - Contains the following datasets:
        -SimLex-999 (SimLex-999: Evaluating Semantic Models with (Genuine) Similarity Estimation. 2014. Felix Hill, Roi Reichart and Anna Korhonen.)
        -MTURK-771 (Guy Halawi, Gideon Dror, Evgeniy Gabrilovich, Yehuda Koren: Large-scale learning of word relatedness with constraints. KDD 2012: 1406-1414)
        -wordsim353 (Eneko Agirre, Enrique Alfonseca, Keith Hall, Jana Kravalova, Marius Pasca, Aitor Soroa, A Study on Similarity and Relatedness Using Distributional and WordNet-based Approaches, In Proceedings of NAACL-HLT 2009.)
        -Stanford Rare Words Dataset (Luong, Minh-Thang  and  Socher, Richard and Manning, Christopher D. Better Word Representations with Recursive Neural Networks for Morphology. 2013)
        -MEN Test Collection (Multimodal Distributional Semantics E. Bruni, N. K. Tran and M. Baroni. Journal of Artificial Intelligence Research 49: 1-47.)
- iksimilarity.py: The main module for synonym support. See the sections "The iksimilarity module" and "An intro demo" for more information.
- modelevaluation.py: Simple Python script to run a word similarity test on model(s) of your choice. You can choose from a list of models currently saved or run the evaluation on all saved models.

## Installation
The synonymdetection package is made available through PyPI:

    pip install synonymdetection

## Notes before diving in
The synonym detection capabilities are made possible through the implementations of Word2Vec and fastText through the [gensim](https://radimrehurek.com/gensim/) Python library. This library is used for training, saving, loading, updating, and actually using the models/vectors. If you installed synonymdetection with pip, gensim (and its dependencies) should've been installed automatically. If not, you can also install gensim with pip:

    pip install gensim

This document describes the Python package itself, but use of this package through InterSystems IRIS does not require an understanding of the package itself. While reading, remember that the descriptions are meant to explain how to use this package from within Python.

The example model that is contained is only to give an idea of how to load and use models, and of the general file layout. It will not provide good measurements of similarity, so don't worry if you play around and the model tells you the most similar word to 'model' is 'for' or some similar oddity. It's also worth nothing that full fledged models are much more involved than the example Word2Vec model that comes prepackaged in the models directory--larger storage size, longer to load into memory for usage, much longer to train etc.

As a final note, when you get to the retraining portion of the demo, if you follow along your outputted value might be different from mine. Every time a model is trained, even on the same corpus, the vectors will be slightly different because the algorithm will initialize each word to a random vector. Besides, the most important part of the model is the general relationship between vectors.

## The iksimilarity module
Most of the functionality is contained in the **IKSimilarityTools** class. This is where you'll find methods for retrieving the most similar words to a provided term, a numerical score (cosine similarity) for similarity between two words etc. For actually using this functionality, you should use either **IKFastTextTools** or **IKWord2VecTools**, both of which extend **IKSimilarityTools**. As is expected, the former is for using fastText models and the latter is for using Word2Vec models, with **IKSimilarityTools** containing the bulk of the shared processes for each. 

There are two other classes in the module: **IKFastTextModeling** and **IKWord2VecModeling**. These classes contain static methods to create (train) and update (retrain) models. The reasoning behind separating out the training/retraining functionality from the implementation functionality was to allow entire "tools" to be instantiated based on given trained models, where the tools can be used solely for similarity detection purposes. The modeling itself (more behind-the-scenes stuff) could then be handled by the modeling classes through static methods.

A "guiding principle" in making this module has been to ensure that one can use it without worrying about everything needed for interfacing with gensim. That is, I think in the end, it should be straightforward to use the iksimilarity module while the module itself handles all of the more intricate interfacing with gensim and the various things needed to train and retrain etc.

## An intro demo

### Instantiating a Word2Vec tool
If you haven't installed the package yet, do so now (see: Installation). Since the package contains an example Word2Vec model, let's see what it looks like to get it up and running. First, we'll import the IKWord2VecTools class from the iksimilarity module, then create an object of the class. To do so, we have to pass in the name of the model that the tool will use for the sake of all calculations.

    >>> from synonymdetection.iksimilarity import IKWord2VecTools as w2v
    >>> demo_tool = w2v(pmodel_name='w2v_example')

Initially, my goal was to load a default model if the specified model wasn't found. Since I wasn't sure that would be the plan (default models), I removed that approach and now just throw an error when someone tries to instantiate an inexistent model. But, a default-model functionality may be something to reconsider for the future.

### Retrieving similar words
We can now use this tool to get information out of the model. One method to do so is most_similar(). Let's say we want to get the 5 most similar words to the word 'model':

    >>> demo_tool.most_similar('model')
    ['Unfortunately,', 'random,', 'contained', 'large', 'and']

If we didn't want 5 terms, we could specify with the num_similar parameter (the default is 5):

    >>> demo_tool.most_similar(term='model', num_similar=10)
    ['Unfortunately,', 'random,', 'contained', 'large', 'and', 'far', 'training', 'not', 'real', 'It']
    >>> demo_tool.most_similar(term='model', num_similar=1)
    Unfortunately,

Note that when we ask for just the single top most similar word, we get a string instead of a list. 

### Measuring similarity
If we want a measure of the similarity between 'model' and 'training', we can use the get_similarity method, passing in the two words we want to compare:

    >>> demo_tool.get_similarity('model', 'training')
    0.060506366

We can check of any words, even if they are not in the model. In those cases, we'll get back -1:

    >>> demo_tool.get_similarity('model', 'cowabunga')
    -1

### Getting a dictionary of synonyms
To utilize this functionality with use_iknow_entities=True, it is necessary to have iknowpy installed. If we had some source text (a corpus or a string), we could get a dictionary containing the top similar words for each word in that text. We could also get the top synonyms for each iKnow entity in the source text passed in, but note that this won't be of much use here because (a) this model wasn't trained on an iKnow preprocessed corpus and (b) it is Word2Vec, so it cannot build a vector for an iKnow entity it has not explicitly seen before (and it will have seen none, other than single-word entities). There are two different methods: synonym_dict_from_string() and synonym_dict_from_file(). The former is used below, and the source text is a free string passed in:
 
    >>> source_text = 'This is an example sentence'
    >>> demo_tool.synonym_dict_from_string(source_text=source_text, use_iknow_entities=False)
    {'This': ['within', 'used', 'corpus.', 'GitHub!', 'has'], 'is': ['in', 'to', 'testing', 'The', 'corpus.'], 'an': ['The', 'large', 'will', 'to', 'real'], 'example': ['contained', 'used', 'has', 'vectors', 'testing']}

Note that since 'sentence' wasn't in the models vocabulary, it was not able to return synonyms for it. We can accomplish the same thing with the path to a file instead of a free string using the synonym_dict_from_file() method. Here, I use the example corpus this model was trained on, which you should also have in your corpora directory. The returned dictionary is too large to print here without making things too dirty, so I abbreviated it.

    >>> source_text = 'corpora/examplecorpus.txt'
    >>> demo_tool.synonym_dict_from_file(source_text=source_text, use_iknow_entities=False)
    {'This': ['within', 'used', 'corpus.', 'GitHub!', 'has'], 'is': ['in', 'to', 'testing', 'The', 'corpus.'], 'an': ['The', 'large', 'will', 'to', 'real'], 'example': ['contained', 'used', 'has', 'vectors', 'testing'], 'It': ['large', 'data', 'essentially', 'corpus.', 'useless'], 'will': ['training', 'any', 'far', 'contained', 'this'], 'not': ['actually', 'Unfortunately,', 'GitHub!', 'for', 'model']
    ...

If we want more or less than 5 top similar terms for the dict, we could specify with a num_similar parameter, similar to most_similar:

    >>> source_text = 'This is an example sentence'
    >>> demo_tool.get_synonym_dict(source_text=source_text, use_iknow_entities=False, num_similar=1)
    {'This': ['within'], 'is': ['in'], 'an': ['The'], 'example': ['contained']}

### Changing models within an existing tool
If we had a second model, lets say w2v_example2, and we wanted to switch to using that for the sake of comparison, we could do so by loading that model into the current tool with load_vectors(). This will do away with the current model and begin using the vectors of the specified model (note that it will only load the vectors, which is quicker and gives us all the functionality we need). Since we *don't* have a second model, it will reject our attempt to switch models and continue with the current model, raising an error in the process so one can handle that if it happens. To keep it clean, I excluded the trace and only copied the error message here:

    >>> demo_tool.load_vectors('w2v_example2')
    FileNotFoundError: Model with name w2v_example2 not found. Continuing use of vectors for currently loaded model (w2v_example)

Important: if we wanted to have both tools going at once, this will not work. We would have to create a second tool the same way we instantiated the first.


### Creating models
Moving away from the \*Tools classes, we can use the \*Modeling classes to train models. A quick example of how one could do this, using the examplecorpus.txt follows. I'll use the fastText modeling as an example so that you could have a fastText model on your own machine to play with but ***note that a fastText model trained on the small examplecorpus.txt will have a size of about 1.2GB on your machine when saved*** <sup>1</sup>. Note also that for demo purposes you ***must set pmin_count=1*** or the training algorithm will only consider words that appear at least 10 times in the corpus, which will leave you with an empty vocabulary on the examplecorpus.txt!

    >>> from iksimilarity import IKFastTextModeling as ftmodeling
    >>> ftmodeling.create_new_model(corpus_path='corpora/examplecorpus.txt', pmodel_name='ft_example', pmin_count=1)
    Building vocabulary...

    Finished building vocabulary.

    Training model...

    Finished training model.

    True

This model will now be saved under the name 'ft_example', available for use.

<sup>1</sup> *This 1.2GB figure is daunting, and for good reason. But a lot of this space seems due to the amount needed in general for a model. I'd say it's akin to renting a storage unit: even if you only have a couple of things to store, the smallest possible unit might still be too big for your stuff. A fastText model trained on a much larger corpus clocks in at about 2.4GB when saved in this format on my machine, so it only doubles in size but explodes much larger in terms of encoded information and performance*

### Retraining a model
For a final display, let's go back to the Word2Vec model. Recall that it couldn't provide us with a similarity for 'model' and 'cowabunga', because it hadn't seen the word cowabunga. We'll retrain with a "corpus" (one sentence) and see what happens after:

    >>> exit()
    $ touch corpora/examplecorpus2.txt
    $ echo "This sentence contains cowabunga to retrain the model" > corpora/examplecorpus2.txt
    $ python3
    >>> from iksimilarity import IKWord2VecModeling as w2v_modeling, IKWord2VecTools as w2v
    >>> w2v_modeling.update_model(corpus_path='corpora/examplecorpus2.txt', pmodel_name='w2v_example')
    >>> demo_tool = w2v(pmodel_name='w2v_example')
    demo_tool.get_similarity('model', 'cowabunga')
    0.0015065705

This shows us that 'cowabunga' has been added to the vocabulary thanks to the retraining!

There may be some updates to come while I fix/change some things in the module. Comments, advice, criticism, errors etc are all welcome, and you can feel free to reach out to me at michael.golden@intersystems.com