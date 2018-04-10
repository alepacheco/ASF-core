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
i O
need O
a O
flight O
from O
memphis B-fromloc.city_name
to O
las B-toloc.city_name
vegas I-toloc.city_name
```

# Usage
### Working demo: https://ofis.justanotherdemo.xyz/
> Code on server in branch: stable

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
  "departure_date": "2018-03-25",
  "departure_time": "",
  "return_date": ""
}
```

# Possible improvements
- Improve char embedding Model
- Split Dataset
- Generate more data
- Add lexicon of cities, dates and times
- Move from BIO labels to BIOES (Begin, Inside, Outside, End, Single)
- Replace multi-digit numbers same as single-digit ones
- Split word before and after digit (ex: $5, 5pm)
- Try glove with 50d
- Explore FastText model
- Reduce out-of-training words

# Credits / Resources
https://arxiv.org/pdf/1511.08308.pdf

https://arxiv.org/pdf/1603.01354.pdf

https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html

https://github.com/guillaumegenthial/sequence_tagging
