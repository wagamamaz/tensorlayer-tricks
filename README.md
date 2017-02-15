# Tricks to use TensorLayer

While research in Deep Learning continues to improve the world, we use a bunch of tricks to use TensorLayer day to day.

Here are a summary of some of the tricks, you can also find some tricks in [FQA](http://tensorlayer.readthedocs.io/en/latest/user/more.html#fqa).

If you find a trick that is particularly useful in practice, please open a Pull Request to add it to the document. If we find it to be reasonable and verified, we will merge it in.

## 1. Installation
 * Don't install via `pip`, `git clone https://github.com/zsdonghao/tensorlayer.git` then copy the `tensorlayer` folder into your project, so you can keep your TL version and edit the source code easily
 * If you want to use `pip` install, then install the master version as TL is growing very fast

## 2. Noise layer
 * set `is_fix` to True, and build different graphs for training and testing by reusing the parameters
 * e.g:
```
def mlp(x, is_train=True, reuse=False):
    with tf.variable_scope("mlp", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in_x = InputLayer(x, name='n_input/x')
  
        net_in = DropoutLayer(net_in, keep=0.8, is_fix=True, is_train=is_train, name='n_in/drop')
        net_h0 = DenseLayer(net_in, n_units=800, act=tf.nn.relu, name='n_h0/dense')

        net_h0 = DropoutLayer(net_h0, keep=0.8, is_fix=True, is_train=is_train, name='n_h0/drop')
        net_h1 = DenseLayer(net_h0, n_units=800, act=tf.nn.relu, name='n_h1/conv2d')

        net_h1 = DropoutLayer(net_h1, keep=0.8, is_fix=True, is_train=is_train, name='n_h1/drop')
        net_ho = DenseLayer(net_h1, n_units=10, act=tf.identity, name='n_ho/dense')

        logits = net_ho.outputs
        net_ho.outputs = tf.nn.sigmoid(net_ho.outputs)
        return net_ho, logits
      
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')

net_train, logits = mlp(x, is_train=True, reuse=False)
net_test, _ = mlp(x, is_train=False, reuse=True)

cost = tl.cost.cross_entropy(logits, y_, name='cost')
```

## 2. Get variables for training
 * use `tl.layers.get_variables_with_name` instead of using `net.all_params`
```
train_vars = tl.layers.get_variables_with_name('mlp', True, True)
train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost, var_list=train_vars)
```
  
## 3. Sentences pre-processing
 * use `tl.nlp.process_sentence` to tokenize the sentences
 * then use `tl.nlp.create_vocab` to create a vocabulary and save as txt file
 * finally use `tl.nlp.Vocabulary` to create a vocabulary object from the txt vocabulary file created by `tl.nlp.create_vocab`

## Links of TensorLayer 
 * [Docs](http://tensorlayer.readthedocs.io/en/latest/)
 * [Github](https://github.com/zsdonghao/tensorlayer)


## Author
 - Zhang Rui
