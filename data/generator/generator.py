import random
import csv
from random import randint

starts = ['find flights', 'get my flights', 'find', '', 'book a flight', 'search', 'flights']
month_lst = ['January', 'Feburary', 'March', 'April', 'May', 'June', 'July',
              'August', 'September', 'October', 'November', 'December']
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

def main():
    data = gen_phrase(10000)
    file = open('test.txt', 'w')
    file.write(gen_trainning_data(data[0], data[1]))
    file.close()


def flatten(x):
    return [item for sublist in x for item in sublist]

def multi_tags(number, tag):
    res = []
    for i in range(number):
        if i == 0:
            res.append('B-' + tag)
        else:
            res.append('I-' + tag)

    return res

def align_data(data):
    spacings = [max([len(seq[i]) for seq in data.values()])
                for i in range(len(data[list(data.keys())[0]]))]
    data_aligned = dict()

    for key, seq in data.items():
        str_aligned = ""
        for token, spacing in zip(seq, spacings):
            str_aligned += token + " " * (spacing - len(token) + 1)

        data_aligned[key] = str_aligned

    return data_aligned

def get_start():
    text = random.choice(starts)
    length = len(text.split(' '))
    return [text, ['O'] * length]

# CVS with lots of cities
with open('./cities.csv', 'rt') as f:
    reader = csv.reader(f)
    cities = flatten(list(reader))

def get_city(tag, label):
    random_city = random.choice(cities)
    num_parts = len(random_city.split(' '))
    texts = tag + ' ' + random_city
    tags = ['O']*len(tag.split(' ')) + multi_tags(num_parts, label)
    return [texts, tags]

def getOrigin():
    return get_city('from', 'from')

def getDestination():
    return get_city('to', 'to')


# Relative dates
relav_dates = [['today', 'tomorrow']]
relav_dates.append(days)
relav_dates.append(list(map(lambda x: 'next ' + x, days)))
relav_dates.append(list(map(lambda x: 'this ' + x, days)))
relav_dates = flatten(relav_dates)

def get_date(starts, tag):
    type = randint(0, 1)
    len_start = 0
    start = random.choice(starts)
    if start != '':
        len_start = len(start.split(' '))

    if type == 0:
        #relative
        choice = random.choice(relav_dates)
        if start != '':
            text = start + ' ' + choice
        else:
            text = choice
        len_text = len(choice.split(' '))
        labels = ['O'] * len_start + multi_tags(len_text, tag)
    else:
        #absolute
        choice = random.choice(month_lst) + ' DIGITDIGIT' + (' th' if randint(0, 1) == 0 else '')
        if start != '':
            text = start + ' ' + choice
        else:
            text = choice
        len_text = len(choice.split(' '))
        labels = ['O'] * len_start + multi_tags(len_text, tag)
    return [text, labels]

def get_departure_date():
    starts = ['', 'on', 'for', 'departing on', 'departing']
    return get_date(starts, 'departure_date')

def get_return_date():
    starts = ['to', 'returning on', 'comming back on', 'returning', 'comming back']
    return get_date(starts, 'arrival_date')

def gen_phrase(nums, log=False):
    sentences_x = []
    sentences_y = []
    for i in range(nums):
        labels = []
        text = []
        sentence = [get_start(), getOrigin(), getDestination(), get_departure_date()]
        if randint(0, 1) == 0:
            sentence.append(get_return_date())
        for part in sentence:
            text.append(part[0])
            labels.append(part[1])

        text = ' '.join(text).split(' ')
        labels = flatten(labels)
        sentences_x.append(text)
        sentences_y.append(labels)

        if log:
            to_print = align_data({"input": text, "output": labels})
            for key, seq in to_print.items():
                print(seq)
            print()

    return [sentences_x, sentences_y]

def gen_trainning_data(X, Y):
    if len(X) != len(Y):
        return 'Error'
    content = ''
    for sentence in range(len(X)):
        for i in range(len(X[sentence])):
            if X[sentence][i] == '':
                continue
            content += X[sentence][i] + ' ' + Y[sentence][i] + '\n'

        content += '\n'
    return content



if __name__  == '__main__':
   main()
