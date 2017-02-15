# Tricks to use TensorLayer

While research in Deep Learning continues to improve the world, we use a bunch of tricks to use TensorLayer day to day.

Here are a summary of some of the tricks, you can also find some tricks in [FQA](http://tensorlayer.readthedocs.io/en/latest/user/more.html#fqa).

If you find a trick that is particularly useful in practice, please open a Pull Request to add it to the document. If we find it to be reasonable and verified, we will merge it in.

## 1. Installation
 * Don't install via `pip`, `git clone https://github.com/zsdonghao/tensorlayer.git` then copy the `tensorlayer` folder into your project, so you can keep your TL version and edit the source code easily
 * If you want to use `pip` install, then install the master version

## 2. Noise layer
 * set `is_fix` to True, and build different graphs for training and testing by reusing (see PTB example)

## 3. Sentences pre-processing
 * use `tl.nlp.process_sentence` to tokenize the sentences
 * then use `tl.nlp.create_vocab` to create a vocabulary and save as txt file
 * finally use `tl.nlp.Vocabulary` to create a vocabulary object from the txt vocabulary file created by `tl.nlp.create_vocab`

## Links of TensorLayer 
 * [Docs](http://tensorlayer.readthedocs.io/en/latest/)
 * [Github](https://github.com/zsdonghao/tensorlayer)


## Author
 - Zhang Rui
