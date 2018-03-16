# Getting started

1. Download the GloVe vectors with

```
make glove
```

Alternatively, you can download them manually [here](https://nlp.stanford.edu/projects/glove/) and update the `glove_filename` entry in `config.py`. You can also choose not to load pretrained word vectors by changing the entry `use_pretrained` to `False` in `model/config.py`.

2. Build the training data, train and evaluate the model with
```
make run
```

# Training Data

The training data must be in the following format (identical to the CoNLL2003 dataset).
A default test file is provided to help you getting started.

```
John B-PER
lives O
in O
New B-LOC
York I-LOC
. O

This O
is O
another O
sentence
```

# Usage

### POST `/parse`

```
curl -X POST "http://localhost:5000/parse" -d "flight to madrid from los angeles for next sunday"
```
#### Response
```
{
  "type": "",
  "departure": "los angeles",
  "destination": "",
  "departureDate": "sunday",
  "returnDate": ""
}
```
