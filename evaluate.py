from itertools import *
import json
import os

import numpy as np

import param_search

DEBUG = False


# Sinustongemisch

def sinustongemisch_dict(frequencies, amplitudes):
    result = dict()
    for idx in range(len(frequencies)):
        result[f'frequency{idx}'] = frequencies[idx]
        result[f'amplitude{idx}'] = amplitudes[idx]
    return result


def sinustongemisch_rename_parmameters(reference_parameters, new_parameters, n_sines):
    frequency_permutations = list(permutations(list(range(n_sines)), n_sines))
    best_permutation = frequency_permutations[0]
    best_deviation = float('inf')
    for perm in frequency_permutations:
        frequency_deviation = 0.
        for idx in range(n_sines):
            ref_frequency = reference_parameters[f'frequency{idx}']
            other_frequency = new_parameters[f'frequency{perm[idx]}']
            frequency_deviation += abs(ref_frequency - other_frequency)
        if frequency_deviation < best_deviation:
            best_deviation = frequency_deviation
            best_permutation = perm

    result = dict()
    for idx in range(n_sines):
        result[f'frequency{idx}'] = new_parameters[f'frequency{best_permutation[idx]}']
        result[f'amplitude{idx}'] = new_parameters[f'amplitude{best_permutation[idx]}']

    return result


def eval_sinustongemisch(n_sines):
    count, iterations, rounds = default_settings()
    result = []
    frequencies = list(combinations(equidistant_frequency(count), n_sines))
    amplitudes = list(combinations(equidistant_amplitude(count), n_sines))
    idx = 0
    for f in frequencies:
        for a in amplitudes:
            print(f'Evaluating {idx + 1} / {len(frequencies) * len(amplitudes)}')
            true_params = sinustongemisch_dict(f, a)
            model, patch_y = param_search.setup_sinustongemisch(true_params)
            _, found_params = param_search.find_parameters(model, patch_y, iterations, rounds)
            found_params = sinustongemisch_rename_parmameters(true_params, found_params, n_sines)
            r = {
                'actual': true_params,
                'predicted': found_params
            }
            result.append(r)
            idx += 1

    debug_str = 'DEBUG' if DEBUG else ''
    with open(f'eval/{debug_str}sinustongemisch_nsines={n_sines}count={count}_iterations={iterations}'
              f'_rounds={rounds}.json', 'w') as f:
        json.dump(result, f, indent=4)


# Amplitude Modulation

def eval_amplitude_modulation():
    count, iterations, rounds = default_settings()
    result = []
    frequencies = list(combinations(equidistant_frequency(count), 2))
    amplitudes = list(combinations(equidistant_amplitude(count), 2))
    idx = 0
    for f in frequencies:
        for a in amplitudes:
            print(f'Evaluating {idx + 1} / {len(frequencies) * len(amplitudes)}')
            true_params = amplitude_modulation_dict(f, a)
            model, patch_y = param_search.setup_amplitude_modulation(true_params)
            _, found_params = param_search.find_parameters(model, patch_y, iterations, rounds)
            r = {
                'actual': true_params,
                'predicted': found_params
            }
            result.append(r)
            idx += 1

    debug_str = 'DEBUG' if DEBUG else ''
    with open(f'eval/{debug_str}amplitude_modulation_count={count}_iterations={iterations}'
              f'_rounds={rounds}.json', 'w') as f:
        json.dump(result, f, indent=4)


def amplitude_modulation_dict(frequencies, amplitudes):
    result = dict()
    result[f'frequency_carrier'] = frequencies[0]
    result[f'amplitude_carrier'] = amplitudes[0]
    result[f'frequency_modulator'] = frequencies[1]
    result[f'amplitude_modulator'] = amplitudes[1]
    return result


# Harmonic

def eval_harmonic(waveshape, n_tones):
    count, iterations, rounds = default_settings()
    result = []
    frequencies = list(combinations(equidistant_frequency(count), n_tones))
    amplitudes = list(combinations(equidistant_amplitude(count), n_tones))
    idx = 0
    for f in frequencies:
        for a in amplitudes:
            print(f'Evaluating {idx + 1} / {len(frequencies) * len(amplitudes)}')
            true_params = sinustongemisch_dict(f, a)
            model, patch_y = param_search.setup_harmonic_tones(waveshape, true_params)
            _, found_params = param_search.find_parameters(model, patch_y, iterations, rounds)
            found_params = sinustongemisch_rename_parmameters(true_params, found_params, n_tones)
            r = {
                'actual': true_params,
                'predicted': found_params
            }
            result.append(r)
            idx += 1

    debug_str = 'DEBUG' if DEBUG else ''
    with open(f'eval/{debug_str}harmonic_waveshape={waveshape}_ntones={n_tones}_count={count}_iterations={iterations}'
              f'_rounds={rounds}.json', 'w') as f:
        json.dump(result, f, indent=4)


# Frequency Modulation

def eval_frequency_modulation():
    count, iterations, rounds = default_settings('frequency modulation')
    result = []
    eqf = equidistant_frequency(count)
    frequencies = list(product(eqf, eqf, eqf))
    amplitudes = equidistant_amplitude(count)
    idx = 0
    for f in frequencies:
        for a in amplitudes:
            print(f'Evaluating {idx + 1} / {len(frequencies) * len(amplitudes)}')
            true_params = frequency_modulation_dict(f, a)
            model, patch_y = param_search.setup_frequency_modulation(true_params)
            _, found_params = param_search.find_parameters(model, patch_y, iterations, rounds)
            r = {
                'actual': true_params,
                'predicted': found_params
            }
            result.append(r)
            idx += 1

    debug_str = 'DEBUG' if DEBUG else ''
    with open(f'eval/{debug_str}frequency_modulation_count={count}_iterations={iterations}'
              f'_rounds={rounds}.json', 'w') as f:
        json.dump(result, f, indent=4)


def frequency_modulation_dict(frequencies, amplitude):
    result = dict()
    result[f'amplitude_carrier'] = amplitude
    result[f'frequency_carrier'] = frequencies[0]
    result[f'frequency_modulator'] = frequencies[1]
    result[f'modulation_depth'] = frequencies[2]
    return result


# Common

def default_settings(synth='default'):
    if DEBUG:
        count = 2
        iterations = 10
        rounds = 1
        return count, iterations, rounds

    if synth == 'default':
        count = 6
        iterations = 500
        rounds = 3
    elif synth == 'frequency modulation':
        count = 5
        iterations = 500
        rounds = 3
    else:
        raise ValueError()

    return count, iterations, rounds


def equidistant_frequency(count):
    start = np.log10(20.)
    end = np.log10(5000.)
    result = np.logspace(start, end, count)
    return result


def equidistant_amplitude(count):
    start_amplitude = 0.05
    end_amplitude = 0.95
    start = np.log10(start_amplitude ** 2)
    end = np.log10(end_amplitude ** 2)
    result = np.logspace(start, end, count)  # power
    result = np.sqrt(result)  # amplitude
    return result


def main():
    # eval_frequency_modulation()
    # eval_amplitude_modulation()
    for waveshape in ['square', 'triangle', 'sawtooth']:
        eval_harmonic(waveshape, 2)


if __name__ == '__main__':
    if not os.path.exists('eval'):
        os.mkdir('eval')
    main()
