from glob import glob
import csv

def red_csv(s):
    input_file = 'counts.csv'
    data = {}
    with open(input_file, 'r') as f:
        true_count = csv.DictReader(f, delimiter=',')
        print(true_count)
        for key in true_count:
            print(key)
            data[key['file']] = key['true_concentration']

    return data

def get_true_concentration(s, data):
    return data[s]

