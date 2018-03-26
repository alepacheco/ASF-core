import http.client
import json
import csv
import re
import parsedatetime
import datetime

def preprocess(sentence):
    encoded = preprocessTimes(sentence)
    return encoded.split(' ')

def getIATA(city_name):
    conn = http.client.HTTPSConnection("www.google.es")
    request = {
      '2': city_name
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
        return None


def preprocessTimes(sentence):
    times = re.finditer(r'[0-9]{2}(?:am|pm)\b', sentence)
    positions = []
    for item in times:
        positions.append(item.start())

    sentence = list(sentence)
    for index, breakPoint in enumerate(positions):
        sentence.insert(breakPoint + 2 + index, ' ')

    return ''.join(sentence)

def parseDates(date):
    pdt = parsedatetime.Calendar()
    timestruct, result = pdt.parse(date)
    if result:
        return datetime.datetime(*timestruct[:3]).strftime("%Y-%m-%d")
    else:
        return None

def validateTime(time):
    if ':' in time or 'pm' in time or 'am' in time:
        return time
    else:
        return None

def parseLabels(sentence, prediction):
    parsed = {
        'type': 'ow_trip',
        'departure': '',
        'destination': '',
        'departureDate': '',
        'returnDate': '',
        'departureTime': ''
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
            parsed['departureDate'] += ' ' + word
            parsed['type'] = 'round_trip'
        elif label.startswith("I-depart_date"):
            parsed['departureDate'] += ' ' + word
            parsed['type'] = 'round_trip'

        elif label.startswith("B-return_date"):
            parsed['returnDate'] += ' ' + word
        elif label.startswith("I-return_date"):
            parsed['returnDate'] += ' ' + word

        elif label.startswith("B-depart_time"):
            parsed['departureDate'] += ' ' + word
            # in case date number gets interpreted as time
            if validateTime(parsed['departureTime']) is None:
                parsed['departureTime'] = word
        elif label.startswith("I-depart_time"):
            parsed['departureDate'] += ' ' + word
            parsed['departureTime'] += ' ' + word

    parsed['departure'] = getIATA(parsed['departure'])
    parsed['destination'] = getIATA(parsed['destination'])
    parsed['departureDate'] = parseDates(parsed['departureDate'])
    parsed['returnDate'] = parseDates(parsed['returnDate'])
    parsed['departureTime'] = validateTime(parsed['departureTime'])

    return parsed


def align_data(data):
    """Given dict with lists, creates aligned strings

    Adapted from Assignment 3 of CS224N

    Args:
        data: (dict) data["x"] = ["I", "love", "you"]
              (dict) data["y"] = ["O", "O", "O"]

    Returns:
        data_aligned: (dict) data_align["x"] = "I love you"
                           data_align["y"] = "O O    O  "

    """
    spacings = [max([len(seq[i]) for seq in data.values()])
                for i in range(len(data[list(data.keys())[0]]))]
    data_aligned = dict()

    # for each entry, create aligned string
    for key, seq in data.items():
        str_aligned = ""
        for token, spacing in zip(seq, spacings):
            str_aligned += token + " " * (spacing - len(token) + 1)

        data_aligned[key] = str_aligned

    return data_aligned
