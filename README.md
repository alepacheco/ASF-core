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
  "departure": "los angeles",
  "destination": "new york",
  "departureDate": "sunday",
  "returnDate": ""
}
```


# Next steps

- [ ] Add fastText model to correct typos
- [ ] Use separate model to format dates in a standardized way
- [ ] Get IATA codes for cities
- [ ] Train model with IATA codes
