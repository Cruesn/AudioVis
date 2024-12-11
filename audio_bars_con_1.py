import shutil

from colorama import Fore, Style, init
import time
import numpy as np
import pyaudiowpatch as pyaudio

# Initialize colorama
init(autoreset=True)
def generate_bands(start_freq, end_freq, num_bands):
    """
    Divides the frequency range between start_freq and end_freq into num_bands equal sized sections.
    """
    step = (end_freq - start_freq) / num_bands
    return [(int(start_freq + i * step), int(start_freq + (i + 1) * step)) for i in range(num_bands)]

def visualize_audio(CHUNK_SIZE=1024):
    with pyaudio.PyAudio() as p:
        try:
            wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError:
            print("WASAPI is not available. Exiting...")
            exit()

        default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
        if not default_speakers["isLoopbackDevice"]:
            for loopback in p.get_loopback_device_info_generator():
                if default_speakers["name"] in loopback["name"]:
                    default_speakers = loopback
                    break
            else:
                print("Loopback device not found. Exiting...")
                exit()

        print(f"Recording from: ({default_speakers['index']}) {default_speakers['name']}")

        def callback(in_data, frame_count, time_info, status):
            try:
                print('\033c', end='')
                data = np.frombuffer(in_data, dtype=np.int16)

                # Perform Fourier transform and get the magnitudes
                fft_result = np.abs(np.fft.fft(data))
                freqs = np.fft.fftfreq(len(fft_result), 1 / default_speakers["defaultSampleRate"])

                # Only consider the positive half of the spectrum
                fft_result = fft_result[:len(fft_result) // 2]
                freqs = freqs[:len(freqs) // 2]

                # Define frequency bands (in Hz)
                bands = generate_bands(0, 6400, 20)

                bar_values = []
                for band_start, band_end in bands:
                    # Determine indices for frequency band
                    band_indices = np.where((freqs >= band_start) & (freqs < band_end))
                    band_amplitude = np.mean(fft_result[band_indices]) if len(band_indices[0]) > 0 else 0
                    bar_values.append(band_amplitude)

                # Normalize and create bars
                max_amplitude = max(bar_values) if bar_values else 1
                bars = [int((value / max_amplitude) * 150) for value in bar_values]
                bars.pop(0)
                print(f"Calculated bars: {bars}")
                terminal_width = shutil.get_terminal_size().columns
                # Display bars with colors
                for bar in bars:
                    if bar < 15:
                        color = Fore.GREEN
                    elif bar < 30:
                        color = Fore.YELLOW
                    else:
                        color = Fore.RED
                    print(f"\r{' ' * terminal_width}", end="") #don't know if this works
                    print(f"\r{color}{'#' * bar} {Style.RESET_ALL}")
                time.sleep(0.05)
                return (in_data, pyaudio.paContinue)
            except Exception as e:
                print(f'an Error occurred: {type(e), e}')

        with p.open(format=pyaudio.paInt16,
                    channels=default_speakers["maxInputChannels"],
                    rate=int(default_speakers["defaultSampleRate"]),
                    frames_per_buffer=CHUNK_SIZE,
                    input=True,
                    input_device_index=default_speakers["index"],
                    stream_callback=callback) as stream:
            print("Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nVisualization stopped.")

if __name__ == "__main__":
    visualize_audio()