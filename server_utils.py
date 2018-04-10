import json
import re
import parsedatetime
import datetime
import urllib.request


def preprocess(sentence):
    encoded = preprocess_times(sentence)
    return encoded.split(' ')

def get_iata(location_name):
    if location_name == '':
        return None
    url = "https://www.edreams.com/travel/service/geo/autocomplete;searchWord=%(DESTINATION)s;departureOrArrival=DEPARTURE;addSearchByCountry=true;addSearchByRegion=true;product=FLIGHT" % {u'DESTINATION': location_name}
    contents = urllib.request.urlopen(url).read().decode("utf-8")
    return json.loads(contents)[0]['iata']


def preprocess_times(sentence):
    """Split times '12pm' to '12 pm' for the model to prosses correctly """

    times = re.finditer(r'[0-9]{2}(?:am|pm)\b', sentence)
    positions = []
    for item in times:
        positions.append(item.start())

    sentence = list(sentence)
    for index, breakPoint in enumerate(positions):
        sentence.insert(breakPoint + 2 + index, ' ')

    return ''.join(sentence)

def parse_dates(date):
    pdt = parsedatetime.Calendar()
    timestruct, result = pdt.parse(date)
    if result:
        return datetime.datetime(*timestruct[:3]).strftime("%Y-%m-%d")
    else:
        return None

def validate_time(time):
    if ':' in time or 'pm' in time or 'am' in time:
        return time
    else:
        return None

def parse_labels(sentence, prediction):
    parsed = {
        'type': 'ow_trip',
        'departure': '',
        'destination': '',
        'departure_date': '',
        'return_date': '',
        'departure_time': ''
    }

    for i in range(len(sentence)):
        label = prediction[i]
        word = sentence[i]
        # TODO we are not using: arrival_date

        if label == "B-to":
            parsed['destination'] = word
        elif label == "I-to":
            parsed['destination'] += ' ' + word
        elif label == "B-from":
            parsed['departure'] = word
        elif label == "I-from":
            parsed['departure'] += ' ' + word
        elif label.startswith("B-departure_date"):
            parsed['departure_date'] += ' ' + word
        elif label.startswith("I-departure_date"):
            parsed['departure_date'] += ' ' + word
        elif label.startswith("B-return_date"):
            parsed['return_date'] += ' ' + word
            parsed['type'] = 'round_trip'
        elif label.startswith("I-return_date"):
            parsed['return_date'] += ' ' + word
            parsed['type'] = 'round_trip'


    parsed['departure'] = get_iata(parsed['departure'])
    parsed['destination'] = get_iata(parsed['destination'])
    parsed['departure_date'] = parse_dates(parsed['departure_date'])
    parsed['return_date'] = parse_dates(parsed['return_date'])
    parsed['departure_time'] = validate_time(parsed['departure_time'])

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
