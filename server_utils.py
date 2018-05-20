import json
import re
import parsedatetime
import datetime
import urllib.request
import urllib.parse
import http.client
from server_config import ServerConfig
from num2words import num2words

def preprocess(sentence):
    encoded = preprocess_times(sentence)
    encoded = preprocess_dates(encoded)
    return encoded.split(' ')

def get_iata_internal(location_name):
    """ Gets iata info from edreams, only works internally """
    if location_name == '':
        return None

    url = "http://geo.odigeo.com/geo-service/geo/v3/autocomplete/findSorted;searchInAllWords=true;departureOrArrival=DEPARTURE;website=GB;multiLanguage=true;addSearchByRegion=true;searchKey=%(DESTINATION)s;locale=en_EN;addSearchByCountry=true;productType=FLIGHT;relatedLocations=true" % {u'DESTINATION': urllib.parse.quote(location_name)}
    try:
        contents = urllib.request.urlopen(url).read().decode("utf-8")
        return json.loads(contents)[0]['iataCode']
    except Exception as e:
        print('get_iata_internal: ', str(e))
        return None

def get_iata_external(city):
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
        return None

def get_iata(location_name):
    if ServerConfig.use_internal_iata:
        return get_iata_internal(location_name)
    return get_iata_external(location_name)

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

def preprocess_dates(sentence):
    """ Cahnges ordinal diogits 24th to wtritten version twenty-four """
    times = re.finditer(r'\d{1,2}(?:st|nd|rd|th)\b', sentence)
    dates = []
    for i in times:
        match = sentence[i.start():i.end()]
        date = ''.join(list(filter(lambda x : x.isdigit(), match)))
        dates.append((match, num2words(int(date), to='ordinal')))

    for date in dates:
        sentence = sentence.replace(date[0], date[1])

    return sentence

def preprocess_number_date(pdt, date):
    """ Convers written numbers to ordinal digits """
    pdt.ptc.small['first'] = 1
    pdt.ptc.small['eleventh'] = 11
    pdt.ptc.small['twenty-first'] = 21
    pdt.ptc.small['thirty-first'] = 31
    pdt.ptc.small['second'] = 2
    pdt.ptc.small['twelfth'] = 12
    pdt.ptc.small['twenty-second'] = 22
    pdt.ptc.small['third'] = 3
    pdt.ptc.small['thirteenth'] = 13
    pdt.ptc.small['twenty-third'] = 23
    pdt.ptc.small['fourth'] = 4
    pdt.ptc.small['fourteenth'] = 14
    pdt.ptc.small['twenty-fourt'] = 24
    pdt.ptc.small['fifth'] = 5
    pdt.ptc.small['fifteenth'] = 15
    pdt.ptc.small['twenty-fifth'] = 25
    pdt.ptc.small['sixth'] = 6
    pdt.ptc.small['sixteenth'] = 16
    pdt.ptc.small['twenty-sixth'] = 26
    pdt.ptc.small['seventh'] = 7
    pdt.ptc.small['seventeenth'] = 17
    pdt.ptc.small['twenty-seven'] = 27
    pdt.ptc.small['eighth'] = 8
    pdt.ptc.small['eighteenth'] = 18
    pdt.ptc.small['twenty-eight'] = 28
    pdt.ptc.small['ninth'] = 9
    pdt.ptc.small['nineteenth'] = 19
    pdt.ptc.small['twenty-ninth'] = 29
    pdt.ptc.small['tenth'] = 10
    pdt.ptc.small['twentieth'] = 20
    pdt.ptc.small['thirtieth'] = 30

    splitted = date.split(' ')
    for idx in range(len(splitted)):
        for i in range(len(splitted),idx,-1):
            analyze = splitted[idx:i]
            try:
                words = ' '.join(analyze)
                x = pdt._convertUnitAsWords(words)
                leading = ' '.join(splitted[0:idx])
                trailing = ' '.join(splitted[i:])
                return leading + ' '+str(x)+'th ' +trailing
            except:
                pass
    return date

def parse_dates(date):
    pdt = parsedatetime.Calendar()
    date = preprocess_number_date(pdt, date)

    timestruct, result = pdt.parse(date)
    if result:
        return datetime.datetime(*timestruct[:3]).strftime(ServerConfig.date_format)
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
        'departure_time': '',
        'raw': prediction
    }

    for i in range(len(sentence)):
        label = prediction[i]
        word = sentence[i]
        # TODO we are not using: arrival_date
        if label == 'B-arrival_date':
            label = 'B-return_date'
        if label == 'I-arrival_date':
            label = 'I-return_date'

        if label == "B-to":
            parsed['destination'] = word
        elif label == "I-to":
            parsed['destination'] += ' ' + word

        elif label == "B-from":
            parsed['departure'] = word
        elif label == "I-from":
            parsed['departure'] += ' ' + word

        elif label == "B-departure_date":
            if parsed['departure_date'] != '' and ServerConfig.assume_dates_order:
                parsed['return_date'] = word
                parsed['type'] = 'round_trip'
            else:
                parsed['departure_date'] += ' ' + word

        elif label == "I-departure_date":
            if parsed['return_date'] != '' and ServerConfig.assume_dates_order:
                parsed['return_date'] += ' ' + word
            else:
                parsed['departure_date'] += ' ' + word

        elif label == "B-return_date":
            parsed['return_date'] += ' ' + word
        elif label == "I-return_date":
            parsed['return_date'] += ' ' + word

    if parsed['return_date'] != '':
        parsed['type'] = 'round_trip'

    parsed['departure'] = get_iata(parsed['departure'])
    parsed['destination'] = get_iata(parsed['destination'])
    parsed['departure_date'] = parse_dates(parsed['departure_date'])
    parsed['return_date'] = parse_dates(parsed['return_date'])
    parsed['departure_time'] = validate_time(parsed['departure_time'])

    if parsed['departure_date'] and parsed['return_date']:
        depart_date = datetime.datetime.strptime(parsed['departure_date'], ServerConfig.date_format)
        return_date = datetime.datetime.strptime(parsed['return_date'], ServerConfig.date_format)
        if depart_date > return_date:
            parsed['departure_date'] = return_date.strftime(ServerConfig.date_format)
            parsed['return_date'] = depart_date.strftime(ServerConfig.date_format)

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
