from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config
from flask import Flask, request
import json

app = Flask(__name__)
config = Config()

model = NERModel(config)
model.build()
model.restore_session(config.dir_model)

test  = CoNLLDataset(config.filename_test, config.processing_word,
                     config.processing_tag, config.max_iter)
model.evaluate(test)

@app.route('/parse', methods=['GET', 'POST'])
def parse():
    if request.method == 'POST':
        data = request.get_data()
        prediction = model.predict(data.strip().split(" "))
        return json.dumps(prediction)
