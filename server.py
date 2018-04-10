from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config
from flask import Flask, request
import json
import server_utils
from flask_sqlalchemy import SQLAlchemy


app = Flask(__name__)
config = Config()
model = NERModel(config)
model.build()
model.restore_session(config.dir_model)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database/test.db'
db = SQLAlchemy(app)
def main():
    db.create_all()
    app.run(debug=True)


class Question(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.String(250), unique=False, nullable=False)

    def __repr__(self):
        return '<Question %r>' % self.question

@app.route('/parse', methods=['POST'])
def parse():
    data = request.get_data()
    sentence = server_utils.preprocess(data.strip().decode())

    db.session.add(Question(question=data.strip().decode()))
    db.session.commit()

    prediction = model.predict(sentence)
    print(prediction)
    parsed = server_utils.parse_labels(sentence, prediction)

    return json.dumps(parsed) + '\n'

@app.route('/data', methods=['GET'])
def data():
   questions = Question.query.all()
   return '\n'.join(str(e) for e in questions)


if __name__  == '__main__':
   main()

from flask import Flask

app = Flask(__name__)


