import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, bilinear, freqz, cheby1, dimpulse, dlti


def design_butterworth_filter(filter_order, cutoff_frequency, sampling_frequency):
    # Design the analog Butterworth filter
    analog_b, analog_a = butter(
        filter_order,
        cutoff_frequency,
        analog=True,
        btype='low'
    )

    # Perform the bilinear transformation
    digital_b, digital_a = bilinear(
        analog_b,
        analog_a,
        sampling_frequency
    )

    return digital_b, digital_a


def design_chebyshev_filter(
    filter_order,
    cutoff_frequency,
    sampling_frequency,
    ripple
):
    # Design the analog Chebyshev Type-I filter
    analog_b, analog_a = cheby1(
        filter_order,
        ripple,
        cutoff_frequency,
        analog=True,
        btype='low'
    )

    # Perform the bilinear transformation
    digital_b, digital_a = bilinear(
        analog_b,
        analog_a,
        sampling_frequency
    )

    return digital_b, digital_a


def plot_filter_response(
    digital_b,
    digital_a,
    sampling_frequency,
    filter_name
):
    # ---------------------------------------------------------
    # 1. Frequency response
    # ---------------------------------------------------------
    frequency, response = freqz(
        digital_b,
        digital_a,
        fs=sampling_frequency,
        worN=4096
    )

    magnitude_response = np.abs(response)

    plt.figure(figsize=(10, 6))
    plt.plot(frequency, magnitude_response)
    plt.title(f'{filter_name} Filter - Magnitude Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.xlim(0, sampling_frequency / 2)
    plt.show()

    # ---------------------------------------------------------
    # 2. Impulse response
    # ---------------------------------------------------------
    system = dlti(digital_b, digital_a)

    time, impulse_response = dimpulse(
        system,
        n=50
    )

    impulse_response = np.squeeze(impulse_response)

    plt.figure(figsize=(10, 6))
    plt.stem(
        time,
        impulse_response
    )

    plt.title(f'{filter_name} Filter - Impulse Response')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()


# =============================================================
# Filter Specifications
# =============================================================

filter_order = 4
cutoff_frequency = 1000       # Hz
sampling_frequency = 8000     # Hz
ripple = 0.5                  # dB


# =============================================================
# Butterworth Filter
# =============================================================

butter_b, butter_a = design_butterworth_filter(
    filter_order,
    cutoff_frequency,
    sampling_frequency
)

print("\nButterworth Filter Coefficients:")
print("Numerator (b):")
print(butter_b)

print("\nDenominator (a):")
print(butter_a)

plot_filter_response(
    butter_b,
    butter_a,
    sampling_frequency,
    "Butterworth"
)


# =============================================================
# Chebyshev Filter
# =============================================================

cheby_b, cheby_a = design_chebyshev_filter(
    filter_order,
    cutoff_frequency,
    sampling_frequency,
    ripple
)

print("\nChebyshev Filter Coefficients:")
print("Numerator (b):")
print(cheby_b)

print("\nDenominator (a):")
print(cheby_a)

plot_filter_response(
    cheby_b,
    cheby_a,
    sampling_frequency,
    "Chebyshev"
)


# =============================================================
# Save Chebyshev Filter Coefficients
# =============================================================

filter_path = "filter_coefficients.txt"

np.savetxt(
    filter_path,
    np.vstack((cheby_b, cheby_a)),
    delimiter=","
)

print(f"\nFilter coefficients saved at: {filter_path}")