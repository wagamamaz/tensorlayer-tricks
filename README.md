# Tricks to use TensorLayer

While research in Deep Learning continues to improve the world, we use a bunch of tricks to use TensorLayer day to day.

Here are a summary of some of the tricks.

If you find a trick that is particularly useful in practice, please open a Pull Request to add it to the document. If we find it to be reasonable and verified, we will merge it in.

* Links 
 * [Docs](http://tensorlayer.readthedocs.io/en/latest/)
 * [Github](https://github.com/zsdonghao/tensorlayer)

* Noise layer
 * set `is_fix` to True, and build different graphs for training and testing by reusing (see PTB example)

* Sentences pre-processing
 * use `tl.nlp.process_sentence` to tokenize the sentences
 * then use `tl.nlp.create_vocab` to create a vocabulary and save as txt file
 * finally use `tl.nlp.Vocabulary` to create a vocabulary object from the txt vocabulary file created by `tl.nlp.create_vocab`
