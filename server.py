from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config
from flask import Flask, request
import json
import server_utils

app = Flask(__name__)
config = Config()
model = NERModel(config)
model.build()
model.restore_session(config.dir_model)

def main():
    app.run(debug=True)


@app.route('/parse', methods=['POST'])
def parse():
    data = request.get_data()
    sentence = data.strip().decode()
    prediction = model.predict(server_utils.preprocess(sentence))

    sentence = sentence.split(' ')

    parsed = server_utils.parseLabels(sentence, prediction)

    # get iatas from cities
    parsed['departure'] = server_utils.getIATA(parsed['departure'])
    parsed['destination'] = server_utils.getIATA(parsed['destination'])
    #parsed['departureDate'] = server_utils.parseDates(parsed['departureDate'])
    #parsed['returnDate'] = server_utils.parseDates(parsed['returnDate'])


    # TODO parse: times
    return json.dumps(prediction) + '\n' +json.dumps(parsed) + '\n'

if __name__ == '__main__':
   main()
