import http.client
import json
import csv
import re
import parsedatetime
import datetime

def preprocess(sentence):
    encoded = encodeDigits(sentence)
    encoded = encodeMonths(encoded)
    encoded = encodeIatas(encoded)
    return encoded.split(' ')


def getIATA(city):
    conn = http.client.HTTPSConnection("www.google.es")
    request = {
      '2': city
    }
    payload = json.dumps({
      '1':[
        {
          '1': 'aa',
          '2': json.dumps(request)
        }
      ]
    })
    headers = {
        'x-gwt-permutation': "C6AE2226F2736ED890C050AF7708C2FD",
        'cache-control': "no-cache",
    }
    conn.request("POST", "/flights/rpc", payload, headers)
    res = conn.getresponse()
    data = json.loads(res.read().decode("utf-8"))
    aiports = json.loads(data['1'][0]['2'])
    try:
        return aiports['3'][0]['2']
    except KeyError:
        return ''

def validateIATA(iata):
    code = iata.strip()
    with open('data/IATAs.csv', 'rt') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if code == row[0]:
                return True
    return False

def encodeDigits(sentence):
    return ''.join(list(map(lambda x: 'DIGIT' if x.isdigit() else x, sentence)))

def encodeMonths(sentence):
    return re.sub(r'(?i)(january|february|march|april|may|june|july|august|september|october|november|december)', 'MONTH', sentence)

def encodeIatas(sentence):
    candidates = re.findall(r'\b[a-zA-Z]{3}\b', sentence)
    iatas = list(filter(validateIATA, candidates))
    if len(iatas) < 1:
        return sentence
    regexExp = re.compile('\\b(' + '|'.join(iatas) + ')\\b')
    return re.sub(regexExp, 'IATA', sentence)

def parseDates(date):
    pdt = parsedatetime.Calendar()
    timestruct, result = pdt.parse(date)
    if result:
        return datetime.datetime(*timestruct[:3]).strftime("%Y-%m-%d")
    else:
        return ''

def parseLabels(sentence, prediction):
    parsed = {
        'type': '',
        'departure': '',
        'destination': '',
        'departureDate': '',
        'departureTime': '',
        'returnDate': ''
    }

    for i in range(len(prediction)):
        label = prediction[i]
        word = sentence[i]

        if label == "B-round_trip" or label == "I-round_trip":
            parsed['type'] = 'round_trip'
        elif label == "B-ow_trip" or label == "I-ow_trip":
            parsed['type'] = 'ow_trip'
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
        elif label.startswith("B-depart_time"):
            parsed['departureTime'] = word
        elif label.startswith("I-depart_time"):
            parsed['departureTime'] += ' ' + word
        elif label.startswith("B-return_date"):
            parsed['returnDate'] = word
        elif label.startswith("I-return_date"):
            parsed['returnDate'] += ' ' + word

    return parsed
