import argparse
import json

import numpy as np


def what_type(filename):
    types = ['sinustongemisch', 'harmonic', 'frequency_modulation', 'amplitude_modulation']
    for t in types:
        if t in filename:
            return t
    raise ValueError()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='json file to be analyzed')
    args = parser.parse_args()

    with open(args.filename, 'r') as f:
        computed = json.load(f)

    assert computed != None

    summary = dict()

    for d in computed:
        actual = d['actual']
        predicted = d['predicted']
        for key in actual:
            if key not in summary:
                summary[key] = dict()
            if actual[key] not in summary[key]:
                summary[key][actual[key]] = []
            summary[key][actual[key]].append(predicted[key])

    if what_type(args.filename) == 'sinustongemisch' or what_type(args.filename) == 'harmonic':
        frequency_keys = []
        amplitude_keys = []
        for key in summary.keys():
            if 'frequency' in key:
                frequency_keys.append(key)
            if 'amplitude' in key:
                amplitude_keys.append(key)
        summary = merge_keys(summary, frequency_keys, 'frequency')
        summary = merge_keys(summary, amplitude_keys, 'amplitude')

    print_statistics(summary)


def merge_keys(dictionary, old_keys, new_key):
    dictionary[new_key] = dict()
    for k in old_keys:
        for true_value in dictionary[k].keys():
            if true_value not in dictionary[new_key]:
                dictionary[new_key][true_value] = []
            dictionary[new_key][true_value].extend(dictionary[k][true_value])
        del dictionary[k]
    return dictionary


def print_statistics(summary):
    print('Name;                    Ref.;    STD;      Q90;      Q95;      Q99;      Max')
    for parameter in summary:
        for reference_value in summary[parameter]:
            result_list = np.abs(np.array(summary[parameter][reference_value]) - reference_value)
            maximum = np.max(result_list)
            q90 = np.percentile(result_list, 90, interpolation='nearest')
            q95 = np.percentile(result_list, 95, interpolation='nearest')
            q99 = np.percentile(result_list, 99, interpolation='nearest')
            standard_dev = np.std(result_list)
            parameter_str = f'{parameter};'.ljust(25)
            reference_str = f'{reference_value:.2f};'.ljust(9)
            print(f'{parameter_str}{reference_str}{standard_dev:.6f}; {q90:.6f}; {q95:.6f}; {q99:.6f}; {maximum:.6f}')


if __name__ == '__main__':
    main()
