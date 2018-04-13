import numpy as np
import os
import re
import csv
import pickle

# shared global variables to be imported from model also
UNK = "$unk$"
NUM = "DIGIT"
MONTH = "$month$"
IATA = "$iata$"
NONE = "O"

class CoNLLDataset(object):
    """Class that iterates over CoNLL Dataset

    __iter__ method yields a tuple (words, tags)
        words: list of raw words
        tags: list of raw tags

    If processing_word and processing_tag are not None,
    optional preprocessing is appplied

    Example:
        ```python
        data = CoNLLDataset(filename)
        for sentence, tags in data:
            pass
        ```

    """
    def __init__(self, filename, processing_word=None, processing_tag=None,
                 max_iter=None):
        """
        Args:
            filename: path to the file
            processing_words: (optional) function that takes a word as input
            processing_tags: (optional) function that takes a tag as input
            max_iter: (optional) max number of sentences to yield

        """
        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.max_iter = max_iter
        self.length = None


    def __iter__(self):
        niter = 0
        with open(self.filename) as f:
            words, tags = [], []
            for line in f:
                line = line.strip()
                if (len(line) == 0 or line.startswith("-DOCSTART-")):
                    if len(words) != 0:
                        niter += 1
                        if self.max_iter is not None and niter > self.max_iter:
                            break
                        yield words, tags
                        words, tags = [], []
                else:
                    ls = line.split(' ')
                    word, tag = ls[0],ls[-1]
                    if self.processing_word is not None:
                        word = self.processing_word(word)
                    if self.processing_tag is not None:
                        tag = self.processing_tag(tag)
                    words += [word]
                    tags += [tag]


    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length


def get_vocabs(datasets):
    """Build vocabulary from an iterable of datasets objects

    Args:
        datasets: a list of dataset objects

    Returns:
        a set of all the words in the dataset

    """
    print("Building vocab...")
    vocab_words = set()
    vocab_tags = set()
    for dataset in datasets:
        for words, tags in dataset:
            vocab_words.update(words)
            vocab_tags.update(tags)
    print("- done. {} tokens".format(len(vocab_words)))
    return vocab_words, vocab_tags


def get_char_vocab(dataset):
    """Build char vocabulary from an iterable of datasets objects

    Args:
        dataset: a iterator yielding tuples (sentence, tags)

    Returns:
        a set of all the characters in the dataset

    """
    vocab_char = set()
    for words, _ in dataset:
        for word in words:
            vocab_char.update(word)

    return vocab_char


def get_glove_vocab(filename):
    """Load vocab from file

    Args:
        filename: path to the glove vectors

    Returns:
        vocab: set() of strings
    """
    print("Building vocab...")
    vocab = set()
    with open(filename, encoding="utf8") as f:
        for line in f:
            word = line.strip().split(' ')[0]
            vocab.add(word)
    print("- done. {} tokens".format(len(vocab)))
    return vocab


def write_vocab(vocab, filename):
    """Writes a vocab to a file

    Writes one word per line.

    Args:
        vocab: iterable that yields word
        filename: path to vocab file

    Returns:
        write a word per line

    """
    print("Writing vocab...")
    with open(filename, "w") as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("- done. {} tokens".format(len(vocab)))


def load_vocab(filename):
    """Loads vocab from a file

    Args:
        filename: (string) the format of the file must be one word per line.

    Returns:
        d: dict[word] = index

    """
    try:
        d = dict()
        with open(filename) as f:
            for idx, word in enumerate(f):
                word = word.strip()
                d[word] = idx

    except IOError:
        raise Exception('Could not find: ' + str(filename) + ' Remember to run python3 build_data.py.')
    return d


def export_trimmed_glove_vectors(vocab, glove_filename, trimmed_filename, dim):
    """Saves glove vectors in numpy array

    Args:
        vocab: dictionary vocab[word] = index
        glove_filename: a path to a glove file
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings

    """
    embeddings = np.zeros([len(vocab), dim])
    with open(glove_filename, encoding="utf8") as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)

    np.savez_compressed(trimmed_filename, embeddings=embeddings)


def get_trimmed_glove_vectors(filename):
    """
    Args:
        filename: path to the npz file

    Returns:
        matrix of embeddings (np array)

    """
    try:
        with np.load(filename) as data:
            return data["embeddings"]

    except IOError:
        raise Exception('Could not find: ' + str(filename) + ' Remember to run python3 build_data.py.')


def get_processing_word(vocab_words=None, vocab_chars=None,
                    lowercase=False, chars=False, allow_unk=True,
                    replace_month=False, replace_digits=True,
                    encode_iatas_bool=False, multiple_digits_same=False):

    """Return lambda function that transform a word (string) into list,
    or tuple of (list, id) of int corresponding to the ids of the word and
    its corresponding characters.

    Args:
        vocab: dict[word] = idx
        multiple_digits_same: Encode 32 and 2 as DIGIT or as DIGITDIGIT
    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)

    """
    def f(word):
        # 0. get chars of words
        if vocab_chars is not None and chars == True:
            char_ids = []
            for char in word:
                if char in vocab_chars:
                    char_ids += [vocab_chars[char]]

        if multiple_digits_same:
            if NUM in word:
                word = NUM
            if replace_digits:
                word = NUM if word.isdigit() else word
        elif replace_digits:
            word = ''.join(list(map(lambda l:(NUM if l.isdigit() else l), word)))
    
        if replace_month:
            word = re.sub(r'(?i)(january|february|march|april|may|june|july|august|september|october|november|december)', MONTH, word)
        if encode_iatas_bool:
            word = encode_iatas(word)
        if lowercase:
            word = word.lower()

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


def encode_iatas(sentence):
    def validateIATA(iata):
        code = iata.strip()
        with open('data/IATAs.csv', 'rt') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if code == row[0]:
                    return True
        return False

    candidates = re.findall(r'\b[a-zA-Z]{3}\b', sentence)
    iatas = list(filter(validateIATA, candidates))
    if len(iatas) < 1:
        return sentence
    regexExp = re.compile('\\b(' + '|'.join(iatas) + ')\\b')
    return re.sub(regexExp, IATA, sentence)



def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids

    Returns:
        a list of list where each sublist has same length

    """
    if nlevels == 1:
        max_length = max(map(lambda x : len(x), sequences))
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


def minibatches(data, minibatch_size):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)

    Yields:
        list of tuples

    """
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


def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}

    Returns:
        tuple: "B", "PER"

    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq, tags):
    """Given a sequence of tags, group entities and their position

    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]

    """
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


def do_train_split(filename_data,filename_train, filename_test, percentage=0.7):
    lines = file_len(filename_data)
    print('We have %d training samples' % lines)
    train_lines = int(percentage*lines)

    with open(filename_data) as f:
        x = f.read()
        x = x.split('\n\n')
        traindata = x[0:train_lines]
        print('Train set is written in %s' % filename_train)
        with open(filename_train, 'w') as output:
            output.write('\n\n'.join(traindata))

        testdata = x[train_lines:-1]
        print('Test set is written in %s' % filename_test)
        with open(filename_test, 'w') as output:
            output.write('\n\n'.join(testdata))

def file_len(fname):
    with open(fname) as f:
        x = f.read()
        x = x.split('\n\n')
        i = len(x)
    return i + 1
def unpickle_atis(filename_atispickle, filename_data):
    train, test, dic = pickle.load(open(filename_atispickle, 'rb'), encoding='latin-1')
    w2idx, ne2idx, labels2idx = dic['words2idx'], dic['tables2idx'], dic['labels2idx']
    idx2w = {w2idx[k]: k for k in w2idx}
    idx2la = {labels2idx[k]: k for k in labels2idx}

    test_x, test_ne, test_label = test
    train_x, train_ne, train_label = train

    wlength = 35
    i = 0
    file = []
    for e in ['train', 'test']:
        for sw, se, sl in zip(eval(e + '_x'), eval(e + '_ne'), eval(e + '_label')):
            #print('WORD'.rjust(wlength), 'LABEL'.rjust(wlength))
            sentence = []
            for wx, la in zip(sw, sl):
                #print(idx2w[wx].rjust(wlength), idx2la[la].rjust(wlength))
                sentence.append(idx2w[wx] + ' ' + idx2la[la])
            sentence = '\n'.join(sentence)
            file.append(sentence)
            i += 1
            #print('\n' + '**' * 30 + '\n')
    with open(filename_data, 'w') as output:
        output.write('\n\n'.join(file))
        #
def refine_classes(filename, classmapping):
    new_lines = []
    with open(filename) as f:
        x = f.readlines()
        for line in x:
            items = line.rstrip().split(' ')
            for maps in ['B-','I-']:
                for item in classmapping:
                    if len(items) > 1 and items[1] == maps+item:
                        if classmapping[item] != 'O':
                            line = line.replace(maps+item, maps+classmapping[item])
                        else:
                            line = line.split(' ')[0] + ' O\n'

            new_lines.append(line)
        #Because of the classes being reduced sometimes we can have the situation that we need to refactor a bit.
        previous_line = None
        for idx, line in enumerate(new_lines):
            tag = line.rstrip().split(' ')
            if previous_line is not None:
                previous_tag = previous_line.rstrip().split(' ')
            if previous_line is not None and len(tag) > 1 and len(previous_tag) > 1 and tag[1] == previous_tag[1]:
                new_lines[idx]= line.replace('B-','I-')
            previous_line = line


    with open(filename, 'w') as output:
        output.write(''.join(new_lines))
