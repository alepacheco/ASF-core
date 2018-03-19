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


# Next steps

- [X] Get IATA codes for cities
- [ ] Fix '10am' interpretation to '10 am'
- [ ] Fix dates digits as time 
- [ ] Remove unused labels
  - [ ] returnDates
