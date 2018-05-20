import os
import logging
from .data_utils import get_trimmed_glove_vectors, load_vocab, \
        get_processing_word
import datetime

class Config():
    def __init__(self, load=True):
        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

        # load if requested (default)
        if load:
            self.load()


    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        # 1. vocabulary
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags  = load_vocab(self.filename_tags)
        self.vocab_chars = load_vocab(self.filename_chars)

        self.nwords     = len(self.vocab_words)
        self.nchars     = len(self.vocab_chars)
        self.ntags      = len(self.vocab_tags)

        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words,
                self.vocab_chars, lowercase=True, chars=self.use_chars)
        self.processing_tag  = get_processing_word(self.vocab_tags,
                lowercase=False, allow_unk=False)

        # 3. get pre-trained embeddings
        self.embeddings = (get_trimmed_glove_vectors(self.filename_trimmed)
                if self.use_pretrained else None)


    # general config
    version = "0.0"
    dir_output = "results/test/" + version +'/'
    dir_model  = dir_output + "model.weights/"
    path_log   = dir_output + "log.txt"

    # embeddings
    dim_word = 300
    dim_char = 100

    # glove files
    filename_glove = "data/glove.6B/glove.6B.{}d.txt".format(dim_word)
    # trimmed embeddings (created from glove_filename with build_data.py)
    filename_trimmed = "data/glove.6B.{}d.trimmed.npz".format(dim_word)
    use_pretrained = True

    # dataset
    filename_data = "data/data.txt"
    filename_dev = filename_test = "data/test.txt"
    filename_train = "data/train.txt"

    max_iter = None # if not None, max number of examples in Dataset

    # vocab (created from dataset with build_data.py)
    filename_words = "data/words.txt"
    filename_tags = "data/tags.txt"
    filename_chars = "data/chars.txt"

    # training
    train_embeddings = False
    nepochs          = 5
    dropout          = 0.5
    batch_size       = 246
    lr               = 0.01
    lr_decay         = 0.9
    clip             = -1 # if negative, no clipping
    nepoch_no_imprv  = 3

    # model hyperparameters
    hidden_size_char = 100 # lstm on chars
    hidden_size_lstm = 100 # lstm on word embeddings

    use_chars = False

    mapping = {
    'arrive_date.date_relative'       :'arrival_date',
    'arrive_date.day_name'            :'arrival_date',
    'arrive_date.day_number'          :'arrival_date',
    'arrive_date.month_name'          :'arrival_date',
    'arrive_date.today_relative'      :'arrival_date',
    'arrive_time.end_time'            :'arrival_date',
    'arrive_time.period_mod'          :'arrival_date',
    'arrive_time.period_of_day'       :'arrival_date',
    'arrive_time.start_time'          :'arrival_date',
    'arrive_time.time'                :'arrival_date',
    'arrive_time.time_relative'       :'arrival_date',
    'depart_date.date_relative'       :'departure_date',
    'depart_date.day_name'            :'departure_date',
    'depart_date.day_number'          :'departure_date',
    'depart_date.month_name'          :'departure_date',
    'depart_date.today_relative'      :'departure_date',
    'depart_date.year'                :'departure_date',
    'depart_time.end_time'            :'departure_date',
    'depart_time.period_mod'          :'departure_date',
    'depart_time.period_of_day'       :'departure_date',
    'depart_time.start_time'          :'departure_date',
    'depart_time.time'                :'departure_date',
    'depart_time.time_relative'       :'departure_date',
    'return_date.date_relative'       :'return_date',
    'return_date.day_name'            :'return_date',
    'return_date.day_number'          :'return_date',
    'return_date.month_name'          :'return_date',
    'return_date.today_relative'      :'return_date',
    'return_time.period_mod'          :'return_date',
    'return_time.period_of_day'       :'return_date',
    'fromloc.airport_code'            :'from',
    'fromloc.airport_name'            :'from',
    'fromloc.city_name'               :'from',
    'fromloc.state_code'              :'from',
    'fromloc.state_name'              :'from',
    'toloc.airport_code'              :'to',
    'toloc.airport_name'              :'to',
    'toloc.city_name'                 :'to',
    'toloc.country_name'              :'to',
    'toloc.state_code'                :'to',
    'toloc.state_name'                :'to',
    'stoploc.airport_code'            :'O',
    'stoploc.airport_name'            :'O',
    'stoploc.city_name'               :'O',
    'stoploc.state_code'              :'O',
    'or'                              :'O',
    'month_name'                      :'O',
    'period_of_day'                   :'O',
    'time'                            :'O',
    'time_relative'                   :'O',
    'today_relative'                  :'O',
    'day_name'                        :'O',
    'day_number'                      :'O',
    'days_code'                       :'O',
    'aircraft_code'                   :'O',
    'airline_code'                    :'O',
    'airline_name'                    :'O',
    'airport_code'                    :'O',
    'airport_name'                    :'O',
    'transport_type'                  :'O',
    'booking_class'                   :'O',
    'city_name'                       :'O',
    'class_type'                      :'O',
    'compartment'                     :'O',
    'connect'                         :'O',
    'cost_relative'                   :'O',
    'economy'                         :'O',
    'fare_amount'                     :'O',
    'fare_basis_code'                 :'O',
    'flight_days'                     :'O',
    'flight_mod'                      :'O',
    'flight_number'                   :'O',
    'flight_stop'                     :'O',
    'flight_time'                     :'O',
    'flight'                          :'O',
    'meal'                            :'O',
    'meal_code'                       :'O',
    'meal_description'                :'O',
    'mod'                             :'O',
    'restriction_code'                :'O',
    'round_trip'                      :'O',
    'state_code'                      :'O',
    'state_name'                      :'O',
    }



def get_logger(filename):
    """Return a logger instance that writes in filename

    Args:
        filename: (string) path to log.txt

    Returns:
        logger: (instance of logger)

    """
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
            '%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)

    return logger
