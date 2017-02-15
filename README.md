# Tricks to use TensorLayer

* Noise layer
 * set `is_fix` to True, and build different graphs for training and testing by reusing (see PTB example)

* Sentences pre-processing
 * use `tl.nlp.process_sentence` to tokenize the sentences
 * then use `tl.nlp.create_vocab` to create a vocabulary and save as txt file
 * finally use `tl.nlp.Vocabulary` to create a vocabulary object from the txt vocabulary file created by `tl.nlp.create_vocab`
