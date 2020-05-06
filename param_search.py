import random
import time
import copy

import torch
from torch import nn
from torch import optim

import matplotlib.pyplot as plt
import numpy as np
import librosa

DEVICE = 'cuda:0'
# DEVICE = 'cpu'
torch.pi = torch.acos(torch.zeros(1)).item() * 2

N_PARALLEL = 1000

SIGNAL_RATE = 44100
SECONDS = 8196. / 44100.

NOTE_FREQ_RANGE = (0., 5000.)
AMPLITUDE_RANGE = (0., 2.)

WHITE_NOISE_AMPLITUDE = 0.0


def random_frequency(tensor_size):
    range_freq = NOTE_FREQ_RANGE[1] - NOTE_FREQ_RANGE[0]
    result = torch.rand(tensor_size, device=DEVICE) * range_freq + NOTE_FREQ_RANGE[0]
    return result


def random_amplitude(tensor_size):
    range_amplitude = AMPLITUDE_RANGE[1] - AMPLITUDE_RANGE[0]
    result = torch.rand(tensor_size, device=DEVICE) * range_amplitude + AMPLITUDE_RANGE[0]
    return result


class SynthBase(nn.Module):
    def __init__(self, n_copies):
        nn.Module.__init__(self)
        self.best_fraction = 0.3
        self.worst_fraction = 0.1
        self.temperature = 0.05
        self.n_copies = n_copies
        self.was_looser = torch.zeros(n_copies, device=DEVICE)

    def use_best(self, loss):
        argmin = loss.argmin()
        for param in self.named_parameters():
            param_values = getattr(self, param[0])[argmin].view(-1, 1)
            setattr(self, param[0], torch.nn.Parameter(param_values, requires_grad=False))

    def best(self, loss):
        argmin = loss.argmin()
        best_params = dict()
        for param in self.named_parameters():
            best_params[param[0]] = getattr(self, param[0])[argmin].item()
        return argmin, loss[argmin].item(), best_params

    def update_param(self, name, old_loss, new_loss, old_values):
        current_param = getattr(self, name)
        old_loss_boundary = (-old_loss).topk(int(self.n_copies * self.best_fraction))[0].min() * -1.

        always_accept = torch.rand(current_param.shape, device=DEVICE) < self.temperature
        keep_unchanged = (old_loss < new_loss) & (~always_accept)
        keep_unchanged |= (old_loss <= old_loss_boundary) & (old_loss < new_loss)
        current_param[keep_unchanged] = old_values[keep_unchanged]

        setattr(self, name, current_param)

    def restart(self):
        for param in self.named_parameters():
            name = param[0]
            self.randomize_param(name)
        self.was_looser = torch.zeros(self.n_copies, device=DEVICE)

    def enable_gradient(self):
        for param in self.named_parameters():
            name = param[0]
            parameter = getattr(self, name)
            parameter.requires_grad = True
            # parameter.register_hook(limit_grad)

    def respawn_loosers(self, loss):
        loosing_threshold = loss.topk(int(self.n_copies * self.worst_fraction))[0].min()
        keep_indices = loss < loosing_threshold
        for param in self.named_parameters():
            name = param[0]
            self.randomize_param(name, keep_indices)

        loosers = ~keep_indices
        self.was_looser[loosers] = 1

    def randomize_param(self, name, keep_indices=None):
        if 'frequency' in name or 'modulation_depth' in name:
            param_values = random_frequency(self.n_copies)
        elif 'amplitude' in name:
            param_values = random_amplitude(self.n_copies)
        else:
            assert False

        if keep_indices is not None:
            old_values = getattr(self, name)
            param_values[keep_indices] = old_values[keep_indices]
        setattr(self, name, torch.nn.Parameter(param_values, requires_grad=False))

    def multiplicative_randomize_param(self, name):
        old_values = getattr(self, name)
        factor = (torch.rand(old_values.shape, device=DEVICE) - 0.5) / 5.
        factor += 1.
        param_values = old_values * factor
        setattr(self, name, torch.nn.Parameter(param_values, requires_grad=False))

    def print_statistics(self):
        for param in self.named_parameters():
            name = param[0]
            values = getattr(self, name)
            minimum = torch.min(values).item()
            maximum = torch.max(values).item()
            values_np = values.detach().cpu().numpy()
            perc1 = np.percentile(values_np, 1, interpolation='nearest')
            perc5 = np.percentile(values_np, 5, interpolation='nearest')
            perc50 = np.percentile(values_np, 50, interpolation='nearest')
            perc95 = np.percentile(values_np, 95, interpolation='nearest')
            perc99 = np.percentile(values_np, 99, interpolation='nearest')

            print(name,
                  f'min={minimum:.3f}',
                  f'quantiles={perc1:.3f} {perc5:.3f} >> {perc50:.3f} << {perc95:.3f} {perc99:.3f}',
                  f'max={maximum:.3f}')

    def print_top_k(self, loss, k):
        print(f'top {k} >>')
        indices = (-loss).topk(k)[1].tolist()
        for idx in indices:
            params = dict()
            for param in self.named_parameters():
                params[param[0]] = getattr(self, param[0])[idx].item()

            print('    ', end='')
            l_string = 'L' if self.was_looser[idx].item() != 0 else ' '
            print(f'{loss[idx].item():.1f}{l_string} || ', end='')
            print_params(params)
        print(f'<< top {k}')


class HarmonicTones(SynthBase):
    def __init__(self, n_copies, n_sines, kth_harmonic):
        SynthBase.__init__(self, n_copies)
        self.n_sines = n_sines
        self.kth_harmonic_amplitude = torch.tensor(kth_harmonic, device=DEVICE) / sum(kth_harmonic)
        self.kth_harmonic_multiple = torch.tensor(np.arange(1, len(kth_harmonic) + 1), device=DEVICE)
        for idx in range(n_sines):
            freq_tensor = random_frequency(n_copies)
            amp_tensor = random_amplitude(n_copies)
            self.register_parameter(f'frequency{idx}', torch.nn.Parameter(freq_tensor, requires_grad=False))
            self.register_parameter(f'amplitude{idx}', torch.nn.Parameter(amp_tensor, requires_grad=False))

    def forward(self, t):
        result = None
        for idx in range(self.n_sines):
            amp = getattr(self, f'amplitude{idx}')
            freq = getattr(self, f'frequency{idx}')
            harmonic_frequencies = self.kth_harmonic_multiple.view(-1, 1) * freq.view(-1, 1, 1)
            cutoff = 11025
            harmonic_frequencies[harmonic_frequencies > cutoff] = 0.
            harmonic_notes = torch.sin(t * harmonic_frequencies * 2. * torch.pi)
            all_harmonics = (harmonic_notes * self.kth_harmonic_amplitude.view(-1, 1)).sum(dim=1)
            temp = amp.view(-1, 1) * all_harmonics
            if result is None:
                result = temp
            else:
                result += temp
        return result


def setup_harmonic_tones(waveshape, parameters=None):
    print('Harmonic tones')
    if parameters is None:
        parameters = {
            'frequency0': 20., 'amplitude0': 0.3
            # 'frequency1': 330., 'amplitude1': 0.2
        }
    print('Actual parameters', parameters)
    n_sines = len(parameters) // 2
    highest_harmonic = 9
    if waveshape == 'square':
        kth_harmonic = [1 / k if k % 2 == 1 else 0. for k in range(1, highest_harmonic)]  # square
    elif waveshape == 'sawtooth':
        kth_harmonic = [1 / k for k in range(1, highest_harmonic)]  # sawtooth
    elif waveshape == 'triangle':
        kth_harmonic = [(-1) ** (k // 2) / (k * k) for k in range(1, highest_harmonic)]  # triangle
    else:
        raise ValueError('No known waveshape type')

    print('k-th harmonic:', kth_harmonic)
    patch_y = synthesize(HarmonicTones(1, n_sines, kth_harmonic), parameters)

    # plt.plot(patch_y[0].cpu().numpy(), label='target')
    # plt.legend()
    # plt.show()

    model = HarmonicTones(N_PARALLEL, n_sines, kth_harmonic)
    return model, patch_y


class SinusTonGemisch(SynthBase):
    def __init__(self, n_copies, n_sines):
        SynthBase.__init__(self, n_copies)
        self.n_sines = n_sines
        for idx in range(n_sines):
            freq_tensor = random_frequency(n_copies)
            amp_tensor = random_amplitude(n_copies)
            self.register_parameter(f'frequency{idx}', torch.nn.Parameter(freq_tensor, requires_grad=False))
            self.register_parameter(f'amplitude{idx}', torch.nn.Parameter(amp_tensor, requires_grad=False))

    def forward(self, t):
        result = None
        for idx in range(self.n_sines):
            amp = getattr(self, f'amplitude{idx}')
            freq = getattr(self, f'frequency{idx}')
            temp = amp.view(-1, 1) * torch.sin(t * freq.view(-1, 1) * 2. * torch.pi)
            if result is None:
                result = temp
            else:
                result += temp
        return result


def setup_sinustongemisch(parameters=None):
    print('Sinustongemisch')
    if parameters is None:
        parameters = {
            'frequency0': 440., 'amplitude0': 0.3,
            'frequency1': 330., 'amplitude1': 0.2,
            'frequency2': 500., 'amplitude2': 0.2
        }
    print('actual parameters', parameters)
    n_sines = len(parameters) // 2
    patch_y = synthesize(SinusTonGemisch(1, n_sines), parameters)
    model = SinusTonGemisch(N_PARALLEL, n_sines)
    return model, patch_y


class RingModulation(SynthBase):
    def __init__(self, n_copies):
        SynthBase.__init__(self, n_copies)
        for idx in range(2):
            freq_tensor = random_frequency(n_copies)
            self.register_parameter(f'frequency{idx}', torch.nn.Parameter(freq_tensor, requires_grad=False))

    def forward(self, t):
        result = None
        for idx in range(2):
            freq = getattr(self, f'frequency{idx}')
            temp = torch.sin(t * freq.view(-1, 1) * 2. * torch.pi)
            if result is None:
                result = temp
            else:
                result *= temp
        return result


def setup_ring_modulation(parameters=None):
    print('Ring Modulation')
    if parameters is None:
        parameters = {
            'frequency0': 440.,
            'frequency1': 100.,
        }
    print('actual parameters', parameters)
    patch_y = synthesize(RingModulation(1), parameters)
    model = RingModulation(N_PARALLEL)

    sndfile = 'snd\\RM_frequency0=440_frequency1=100.wav'
    maxmsp, sr = librosa.load(sndfile, sr=44100)
    assert sr == 44100
    maxmsp = torch.from_numpy(maxmsp).to(DEVICE).view(1, -1)
    assert patch_y.shape == maxmsp.shape
    print(sndfile)

    return model, patch_y, maxmsp


class AmplitudeModulation(SynthBase):
    def __init__(self, n_copies):
        SynthBase.__init__(self, n_copies)
        self.register_parameter(f'frequency_carrier',
                                torch.nn.Parameter(random_frequency(n_copies), requires_grad=False))
        self.register_parameter(f'amplitude_carrier',
                                torch.nn.Parameter(random_amplitude(n_copies), requires_grad=False))
        self.register_parameter(f'frequency_modulator',
                                torch.nn.Parameter(random_frequency(n_copies), requires_grad=False))
        self.register_parameter(f'amplitude_modulator',
                                torch.nn.Parameter(random_amplitude(n_copies), requires_grad=False))

    def forward(self, t):
        modulator = torch.sin(t * self.frequency_modulator.view(-1, 1) * 2. * torch.pi)
        amplitude = self.amplitude_carrier.view(-1, 1) + self.amplitude_modulator.view(-1, 1) * modulator
        carrier = torch.sin(t * self.frequency_carrier.view(-1, 1) * 2. * torch.pi)
        return amplitude * carrier


def setup_amplitude_modulation(parameters=None):
    print('Amplitude Modulation')
    if parameters is None:
        parameters = {
            'frequency_carrier': 440.,
            'amplitude_carrier': 0.5,
            'frequency_modulator': 100.,
            'amplitude_modulator': 0.3
        }
    print('Actual:', parameters)
    patch_y = synthesize(AmplitudeModulation(1), parameters)

    model = AmplitudeModulation(N_PARALLEL)
    return model, patch_y


class FrequencyModulation(SynthBase):
    def __init__(self, n_copies):
        SynthBase.__init__(self, n_copies)
        self.register_parameter(f'frequency_carrier',
                                torch.nn.Parameter(random_frequency(n_copies), requires_grad=False))
        self.register_parameter(f'amplitude_carrier',
                                torch.nn.Parameter(random_amplitude(n_copies), requires_grad=False))
        self.register_parameter(f'frequency_modulator',
                                torch.nn.Parameter(random_frequency(n_copies), requires_grad=False))
        self.register_parameter(f'modulation_depth',
                                torch.nn.Parameter(random_frequency(n_copies), requires_grad=False))

    def forward(self, t):
        modulator = self.modulation_depth.view(-1, 1) * torch.sin(
            t * self.frequency_modulator.view(-1, 1) * 2. * torch.pi)
        frequency = self.frequency_carrier.view(-1, 1) + modulator
        dx = t[1] - t[0]
        phase = torch.cumsum(frequency * 2 * torch.pi * dx, dim=1)
        return self.amplitude_carrier.view(-1, 1) * torch.sin(phase)


def setup_frequency_modulation(parameters=None):
    print('Frequency Modulation')
    if parameters is None:
        parameters = {
            'frequency_carrier': 440.,
            'amplitude_carrier': 0.6,
            'frequency_modulator': 100.,
            'modulation_depth': 200.
        }
    print(parameters)
    patch_y = synthesize(FrequencyModulation(1), parameters)
    model = FrequencyModulation(N_PARALLEL)
    return model, patch_y


def synthesize(synth, param_dict):
    for key in param_dict:
        setattr(synth, key, torch.nn.Parameter(torch.tensor(param_dict[key]), requires_grad=False))

    synth.to(DEVICE)
    n_samples = int(SIGNAL_RATE * SECONDS)
    t = torch.linspace(0, SECONDS, n_samples, device=DEVICE)
    t += random.random() * 2.  # random phase
    output = synth(t) + torch.rand(t.shape, device=DEVICE) * WHITE_NOISE_AMPLITUDE
    return output


def make_spectrogram(audio):
    fft = torch.rfft(audio, signal_ndim=1)
    fft_squared = fft * fft
    return torch.sqrt(fft_squared[..., -1] + fft_squared[..., -2]).float()


def compute_loss(output, patch_spectrogram):
    output_spectrogram = make_spectrogram(output)

    diff = output_spectrogram - patch_spectrogram
    loss = torch.sum(diff * diff, dim=1)

    # loss = torch.abs(output_spectrogram - patch_spectrogram).sum(dim=1)
    return loss


def print_params(dict):
    for key in dict:
        print(key, end='')
        print(f': {dict[key]:.3f}, ', end='')
    print()


def main():
    # model, patch_y = setup_sinustongemisch()
    # model, patch_y, maxmsp = setup_ring_modulation()
    # model, patch_y = setup_amplitude_modulation()
    model, patch_y = setup_frequency_modulation()
    # model, patch_y = setup_harmonic_tones('square')
    # model.print_statistics()

    ANALYZE_MAX = False
    if ANALYZE_MAX:
        print('Using WAV file from MAX/MSP')
        patch_y = maxmsp
    else:
        print('Synthesized in python')

    best = find_parameters(model, patch_y, iterations=500, restart=3)
    print(f'FINAL RESULT (loss = {best[0]:.1f})')
    print(best[1])


def improve(model, patch_spectrogram):
    print('Improving with gradient')
    model.enable_gradient()

    n_samples = int(SIGNAL_RATE * SECONDS)
    t = torch.linspace(0, SECONDS, n_samples, device=DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    output = model(t)
    best_loss = compute_loss(output, patch_spectrogram)
    best_model = copy.deepcopy(model)
    improvement = False
    for idx in range(1, 100000):
        optimizer.zero_grad()
        output = model(t)
        loss = compute_loss(output, patch_spectrogram)
        # print(idx, min(loss).item())
        # loss.backward(torch.full((N_PARALLEL,), 1, dtype=float).to(DEVICE))
        loss.backward()
        optimizer.step()

        if loss < best_loss - 1.:
            improvement = True
            best_loss = loss
            best_model = copy.deepcopy(model)

        if idx % 100 == 0:
            if not improvement:
                print(f'No more improvement @{idx}')
                break
            else:
                improvement = False
    return best_model, best_loss


def find_parameters(model, patch_y, iterations, restart):
    patch_spectrogram = make_spectrogram(patch_y)
    best = None
    best_model = None
    for round_idx in range(restart):
        print(f'/// Round {round_idx} \\\\\\')
        model.restart()
        model, loss = run_round(model, patch_spectrogram, iterations)
        if best is None:
            best = model.best(loss)
            best_model = copy.deepcopy(model)
        if model.best(loss)[1] < best[1]:
            best = model.best(loss)
            best_model = copy.deepcopy(model)

    model = copy.deepcopy(best_model)
    n_samples = int(SIGNAL_RATE * SECONDS)
    t = torch.linspace(0, SECONDS, n_samples, device=DEVICE)
    output = model(t)
    loss = compute_loss(output, patch_spectrogram)
    model.use_best(loss)
    model, loss = improve(model, patch_spectrogram)
    if model.best(loss)[1] < best[1]:
        best = model.best(loss)

    return best[1], best[2]


def run_round(model, patch_spectrogram, iterations, verbose=False):
    n_samples = int(SIGNAL_RATE * SECONDS)
    t = torch.linspace(0, SECONDS, n_samples, device=DEVICE)
    param_names = [param[0] for param in model.named_parameters()]
    for idx in range(1, iterations + 1):
        param_name = random.choice(param_names)
        old_param = getattr(model, param_name)

        output = model(t)
        old_loss = compute_loss(output, patch_spectrogram)

        if random.choice([True, False]):
            model.multiplicative_randomize_param(param_name)
        else:
            model.randomize_param(param_name)

        model.to(DEVICE)
        output = model(t)
        loss = compute_loss(output, patch_spectrogram)
        model.update_param(param_name, old_loss, loss, old_param)

        output = model(t)
        loss = compute_loss(output, patch_spectrogram)
        model.respawn_loosers(loss)
        model.to(DEVICE)

        if idx % (iterations // 3) == 0:
            output = model(t)
            loss = compute_loss(output, patch_spectrogram)
            argmin, best_loss, best_params = model.best(loss)
            print(f'{best_loss:5.1f} || ', end='')
            print_params(best_params)
            if verbose:
                model.print_top_k(loss, 5)

            # if idx % 5000 == 0:
            #     plt.plot(output[argmin].detach().cpu().numpy(), label='output')
            #     plt.plot(patch_y[0].cpu().numpy(), label='target')
            #     plt.legend()
            #     plt.show()

    return model, loss


if __name__ == '__main__':
    main()
