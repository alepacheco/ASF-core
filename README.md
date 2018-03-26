# Model overview

![](https://docs.google.com/drawings/d/e/2PACX-1vRsJ0BZjt8btq-1VmAa9Y3MU3cuxKc9FSrBrS5k2nctFqdMmNc-h1gFQiVYO8Fph8J8s_MmwFtGxRKS/pub?w=718&amp;h=656)

# Getting started

1. Download the GloVe vectors with

```
make glove
```

2. Build the training data, train and evaluate the model with
```
make run
```

# Training Data
The training data must be in the IOB format

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

### Start the server
```bash
python3 server.py
```

### POST `/parse`

```sh
curl -X POST "http://localhost:5000/parse" -d "flight to new york from los angeles for next sunday"
```
> Response
```json
{
  "type": "",
  "departure": "LAX",
  "destination": "NYC",
  "departureDate": "2018-03-25",
  "departureTime": "",
  "returnDate": ""
}
```


# Credits
https://arxiv.org/pdf/1511.08308.pdf
https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html
https://github.com/guillaumegenthial/sequence_tagging
