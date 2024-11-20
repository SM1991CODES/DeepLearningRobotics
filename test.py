import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    f = 50
    T = 0.02
    fs = f * 32
    ts = 1 / fs
    t = np.arange(0, 3*T, ts)

    sines = 5 * np.sin(2 * np.pi * f * t) + 3.5 * np.sin(2 * np.pi * 3 * f * t)
    plt.plot(t, sines, color="r")

    fft = np.fft.fft(sines)
    fft_mag = np.abs(fft)
    fft_phase = np.angle(fft)

    bins = np.arange(0, len(fft))
    print(f"Freq each bin -> {fs / len(bins)}")

    fig, axs = plt.subplots(1, 3)
    axs[0].plot(t, sines, label="sines", color="r")
    axs[1].plot(bins, fft_mag, label="FFT_mag", color="g")
    axs[2].plot(bins, fft_phase, label="FFT_phase", color="b")
    plt.show()

    max_freq_bin = np.argmax(fft_mag)
    freq_per_bin = fs / len(bins)  # NOTE: this is how the whole freq is split
    print(f"Highest freq -> {max_freq_bin * freq_per_bin} Hz")

    print("Done")


