import tensorflow as tf
import numpy as np
import os

UNK = "$UNK$"
NUM = "$NUM$"
NONE = "O"

def get_trimmed_glove_vectors():
    filename = "data/glove.6B.300d.trimmed.npz"
    with np.load(filename) as data:
        return data["embeddings"]

def load_vocab(filename):
    d = dict()
    with open(filename) as f:
        for idx, word in enumerate(f):
            word = word.strip()
            d[word] = idx
    return d

def get_processing_word(vocab_words=None, vocab_chars=None,
                    lowercase=False, chars=False, allow_unk=True):
    def f(word):
        # 0. get chars of words
        if vocab_chars is not None and chars == True:
            char_ids = []
            for char in word:
                # ignore chars out of vocabulary
                if char in vocab_chars:
                    char_ids += [vocab_chars[char]]

        # 1. preprocess word
        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = NUM

        # 2. get id of word
        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
            else:
                if allow_unk:
                    word = vocab_words[UNK]
                else:
                    raise Exception("Unknow key is not allowed. Check that "\
                                    "your vocab (tags?) is correct")

        # 3. return tuple char ids, word id
        if vocab_chars is not None and chars == True:
            return char_ids, word
        else:
            return word

    return f

def save_session():
    """Saves session = weights"""
    if not os.path.exists(dir_model):
        os.makedirs(dir_model)
    saver.save(sess, dir_model)

def minibatches(data, minibatch_size):
    x_batch, y_batch = [], []
    for (x, y) in data:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []

        if type(x[0]) == tuple:
            x = zip(*x)
        x_batch += [x]
        y_batch += [y]

    if len(x_batch) != 0:
        yield x_batch, y_batch

def _pad_sequences(sequences, pad_tok, max_length):
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):

    if nlevels == 1:
        max_length = max(map(lambda x : len(x), sequences)) # ffffffffffffffffff
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                            pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq))
                               for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x : len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded,
                [pad_tok]*max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0,
                max_length_sentence)

    return sequence_padded, sequence_length


def get_feed_dict(words, _labels=None, _lr=None, _dropout=None):

    _char_ids, _word_ids = zip(*words)
    _word_ids, _sequence_lengths = pad_sequences(_word_ids, 0)
    _char_ids, _word_lengths = pad_sequences(_char_ids, pad_tok=0,
        nlevels=2)

    # build feed dictionary
    feed = {
        word_ids: _word_ids,
        sequence_lengths: _sequence_lengths
    }

    if use_chars:
        feed[char_ids] = _char_ids
        feed[word_lengths] = _word_lengths

    if _labels is not None:
        _labels, _ = pad_sequences(_labels, 0)
        feed[labels] = _labels

    if lr is not None:
        feed[lr] = _lr

    if dropout is not None:
        feed[dropout] = _dropout

    return feed, _sequence_lengths


def run_evaluate(test):
    accs = []
    correct_preds, total_correct, total_preds = 0., 0., 0.
    for words, labels in minibatches(test, batch_size):
        labels_pred, sequence_lengths = predict_batch(words)

        for lab, lab_pred, length in zip(labels, labels_pred,
                                         sequence_lengths):
            lab      = lab[:length]
            lab_pred = lab_pred[:length]
            accs    += [a==b for (a, b) in zip(lab, lab_pred)]

            lab_chunks      = set(get_chunks(lab, vocab_tags))
            lab_pred_chunks = set(get_chunks(lab_pred, vocab_tags))

            correct_preds += len(lab_chunks & lab_pred_chunks)
            total_preds   += len(lab_pred_chunks)
            total_correct += len(lab_chunks)

    p   = correct_preds / total_preds if correct_preds > 0 else 0
    r   = correct_preds / total_correct if correct_preds > 0 else 0
    f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
    acc = np.mean(accs)

    return {"acc": 100*acc, "f1": 100*f1}

def predict_batch(words):
    fd, sequence_lengths = get_feed_dict(words, _dropout=1.0)

    # get tag scores and transition params of CRF
    viterbi_sequences = []
    logits, trans_params = sess.run(
            [logits, trans_params], feed_dict=fd)

    # iterate over the sentences because no batching in vitervi_decode
    for logit, sequence_length in zip(logits, sequence_lengths):
        logit = logit[:sequence_length] # keep only the valid steps
        viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                logit, trans_params)
        viterbi_sequences += [viterbi_seq]

    return viterbi_sequences, sequence_lengths


def get_chunks(seq, tags):
    default = tags[NONE]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks


def run_epoch(train, dev, epoch):
    nbatches = (len(train) + batch_size - 1) // batch_size

    # iterate over dataset
    for i, (words, labels) in enumerate(minibatches(train, batch_size)):
        fd, _ = get_feed_dict(words, labels, lr, 0.5) # dropout

        _, train_loss, summary = sess.run([train_op, loss], feed_dict=fd)

    metrics = run_evaluate(dev)
    msg = " - ".join(["{} {:04.2f}".format(k, v)
            for k, v in metrics.items()])
    print(msg)

    return metrics["f1"]

def trainModel(train, dev):
    best_score = 0
    nepoch_no_imprv = 0

    for epoch in range(nepochs):
        print("Epoch {:} out of {:}".format(epoch + 1,
                    nepochs))

        score = run_epoch(train, dev, epoch)
        lr *= lr_decay # decay learning rate

        # early stopping and saving best parameters
        if score >= best_score:
            nepoch_no_imprv = 0
            save_session()
            best_score = score
            print("- new best score!")
        else:
            nepoch_no_imprv += 1
            if nepoch_no_imprv >= 3:
                print("- early stopping {} epochs without "\
                        "improvement".format(nepoch_no_imprv))
                break



# training
train_embeddings = False
nepochs          = 15
dropout          = 0.5
batch_size       = 20
lr_method        = "adam"
lr               = 0.001
lr_decay         = 0.9
clip             = -1 # if negative, no clipping
nepoch_no_imprv  = 3

# model hyperparameters
hidden_size_char = 100 # lstm on chars
hidden_size_lstm = 300 # lstm on word embeddings
use_crf = True # if crf, training is 1.7x slower on CPU
use_chars = True # if char embedding, training is 3.5x slower on CPU


filename_words = "data/words.txt"
filename_tags = "data/tags.txt"
filename_chars = "data/chars.txt"
dim_word = 300
dim_char = 100
vocab_words = load_vocab(filename_words)
vocab_tags = load_vocab(filename_tags)
vocab_chars = load_vocab(filename_chars)

nwords = len(vocab_words)
nchars = len(vocab_chars)
ntags = len(vocab_tags)

word_ids = tf.placeholder(tf.int32, shape=[None, None])

sequence_lengths = tf.placeholder(tf.int32, shape=[None])
char_ids = tf.placeholder(tf.int32, shape=[None, None, None])
word_lengths = tf.placeholder(tf.int32, shape=[None, None])
labels = tf.placeholder(tf.int32, shape=[None, None],
                name="labels")
with tf.variable_scope("words"):
    L = tf.Variable(get_trimmed_glove_vectors(), dtype=tf.float32, trainable=False)
    word_embeddings = tf.nn.embedding_lookup(L, word_ids)




# get char embeddings matrix
with tf.variable_scope("chars"):
    # get char embeddings matrix
    _char_embeddings = tf.get_variable(
            name="_char_embeddings",
            dtype=tf.float32,
            shape=[nchars, dim_char])
    char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
            char_ids, name="char_embeddings")

    # put the time dimension on axis=1
    s = tf.shape(char_embeddings)
    char_embeddings = tf.reshape(char_embeddings,
            shape=[s[0]*s[1], s[-2], dim_char])
    word_lengths = tf.reshape(word_lengths, shape=[s[0]*s[1]])

    # bi lstm on chars
    cell_fw = tf.contrib.rnn.LSTMCell(hidden_size_char,
            state_is_tuple=True)
    cell_bw = tf.contrib.rnn.LSTMCell(hidden_size_char,
            state_is_tuple=True)
    _output = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, char_embeddings,
            sequence_length=word_lengths, dtype=tf.float32)

    # read and concat output
    _, ((_, output_fw), (_, output_bw)) = _output
    output = tf.concat([output_fw, output_bw], axis=-1)

    # shape = (batch size, max sentence length, char hidden size)
    output = tf.reshape(output,
            shape=[s[0], s[1], 2*hidden_size_char])
    word_embeddings = tf.concat([word_embeddings, output], axis=-1)

word_embeddings =  tf.nn.dropout(word_embeddings, dropout)

# lo de arrriba era el embedding y la bi-lstm para el 'embedding' de los characters
# now the real bi-lstm Model
with tf.variable_scope("bi-lstm"):
    cell_fw = tf.contrib.rnn.LSTMCell(hidden_size_lstm)
    cell_bw = tf.contrib.rnn.LSTMCell(hidden_size_lstm)

    (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw,
        cell_bw, word_embeddings, sequence_length=sequence_lengths,
        dtype=tf.float32)

    output = tf.concat([output_fw, output_bw], axis=-1)
    output = tf.nn.dropout(output, dropout)

with tf.variable_scope("proj"):
    W = tf.get_variable("W", dtype=tf.float32,
            shape=[2*hidden_size_lstm, ntags])

    b = tf.get_variable("b", shape=[ntags],
            dtype=tf.float32, initializer=tf.zeros_initializer())

    nsteps = tf.shape(output)[1]
    output = tf.reshape(output, [-1, 2*hidden_size_lstm])
    pred = tf.matmul(output, W) + b
    logits = tf.reshape(pred, [-1, nsteps, ntags])


# add loss
log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
        logits, labels, sequence_lengths)
trans_params = trans_params # need to evaluate it for decoding
loss = tf.reduce_mean(-log_likelihood)


optimizer = tf.train.AdamOptimizer(lr)
train_op = optimizer.minimize(loss)

# init stuff

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

# prepare training data
from model.data_utils import CoNLLDataset
filename_dev = filename_test = filename_train = "data/test.txt"
max_iter = None # if not None, max number of examples in Dataset

processing_word = get_processing_word(vocab_words,
        vocab_chars, lowercase=True, chars=use_chars)
processing_tag  = get_processing_word(vocab_tags,
        lowercase=False, allow_unk=False)

dev   = CoNLLDataset(filename_dev, processing_word,
                     processing_tag, max_iter)
train = CoNLLDataset(filename_train, processing_word,
                     processing_tag, max_iter)


trainModel(train, dev)
