from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config
from flask import Flask, request
from num2words import num2words
import json

app = Flask(__name__)
config = Config()
model = NERModel(config)
model.build()
model.restore_session(config.dir_model)

def main():
    app.run(debug=True)

def digits2words(sentence):
    return

def get_digits1(text):
    char = ""
    for i in range(len(text)):
        if text[i].isdigit():
            number = text[i]

    return c


@app.route('/parse', methods=['POST'])
def parse():
    data = request.get_data()
    sentence = data.strip().decode().split(" ")
    # TODO preprosessing, typo-fixing, lowercase, digits to words
    prediction = model.predict(sentence)

    parsed = {
        'type': '',
        'departure': '',
        'destination': '',
        'departureDate': '',
        'returnDate': ''
    }

    for i in range(len(prediction)):
        label = prediction[i]
        word = sentence[i]

        if label == "B-round_trip":
            parsed['type'] = word
        elif label == "I-round_trip":
            parsed['type'] += ' ' + word
        elif label == "B-toloc.city_name":
            parsed['destination'] = word
        elif label == "I-toloc.city_name":
            parsed['destination'] += ' ' + word
        elif label == "B-fromloc.city_name":
            parsed['departure'] = word
        elif label == "I-fromloc.city_name":
            parsed['departure'] += ' ' + word
        elif label.startswith("B-depart_date"):
            parsed['departureDate'] = word
        elif label.startswith("I-depart_date"):
            parsed['departureDate'] += ' ' + word
        elif label.startswith("B-return_date"):
            parsed['returnDate'] = word
        elif label.startswith("I-return_date"):
            parsed['returnDate'] += ' ' + word

    return json.dumps(prediction) + '\n' + json.dumps(parsed) + '\n'



# type
# departure iata: fix typos, find iata
# destination Same above
#

if __name__ == '__main__':
   main()
