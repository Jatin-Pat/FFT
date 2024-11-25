import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cv2
import argparse
import time

class DiscreteFourierTransform:
    def __init__(self, image):
        self.original_image = image
        self.resized_image = self.resize_image(image)
        
    def resize_image(self, image):
        row, col = image.shape
        resized_row, resized_col = 2**int(np.ceil(np.log2(row))), 2**int(np.ceil(np.log2(col)))
        
        padded_image = np.zeros((resized_row, resized_col), dtype=np.uint8)
        padded_image[:row, :col] = image
        
        return padded_image 
    
    def dft(self, signal):
        N = len(signal)
        X = np.zeros(N, dtype=np.complex128)
        
        for k in range(N):
            for n in range(N):
                X[k] += signal[n] * np.exp(-2j * np.pi * k * n / N)
                
        return X
    
    def idft(self, signal):
        N = len(signal)
        x = np.zeros(N, dtype=np.complex128)
        
        for n in range(N):
            for k in range(N):
                x[n] += signal[k] * np.exp(2j * np.pi * k * n / N)
                
            x[n] /= N
                
        return x
    
    def fft(self, signal, size_threshold=8):
        N = len(signal)
        
        if N <= size_threshold:
            return self.dft(signal)
        
        even = self.fft(signal[0::2])
        odd = self.fft(signal[1::2])
        
        coeff = np.exp(-2j * np.pi * np.arange(N) / N)
        
        return np.concatenate([even + coeff[:N//2] * odd, even + coeff[N//2:] * odd])  
    
    def ifft(self, signal, size_threshold=8):
        N = len(signal)
        
        if N <= size_threshold:
            return self.idft(signal)
        
        even = self.ifft(signal[0::2])
        odd = self.ifft(signal[1::2])
        
        coeff = np.exp(2j * np.pi * np.arange(N) / N)
        
        return np.concatenate([even + coeff[:N//2] * odd, even + coeff[N//2:] * odd])
    
    def fft_2d(self):
        row, col = self.resized_image.shape
        
        row_transformed_image = np.zeros((row, col), dtype=np.complex128)
        for i in range(row):
            row_transformed_image[i, :] = self.fft(self.resized_image[i, :])
        
        col_transformed_image = np.zeros((row, col), dtype=np.complex128)
        for j in range(col):
            col_transformed_image[:, j] = self.fft(row_transformed_image[:, j])
        
        return col_transformed_image
    
    def ifft_2d(self, transformed_image):
        row, col = transformed_image.shape
        
        row_transformed_image = np.zeros((row, col), dtype=np.complex128)
        for i in range(row):
            row_transformed_image[i, :] = self.ifft(transformed_image[i, :])
        
        col_transformed_image = np.zeros((row, col), dtype=np.complex128)
        for j in range(col):
            col_transformed_image[:, j] = self.ifft(row_transformed_image[:, j])
        
        return col_transformed_image
    
    def plot_compression(self):
        frequency_domain = self.fft_2d()
        magnitude = np.abs(frequency_domain)
        compression_levels = [1.0, 0.5, 0.1, 0.05, 0.01, 0.001]
        compressed_images = []

        for level in compression_levels:
            threshold = np.quantile(magnitude, 1 - level)
            compressed_frequency = frequency_domain * (magnitude >= threshold)
            compressed_image = self.ifft_2d(compressed_frequency)
            compressed_image = np.real(compressed_image[:self.original_image.shape[0], :self.original_image.shape[1]])
            compressed_images.append(compressed_image)
        
        fig, axs = plt.subplots(2, 3, figsize=(12, 8))
        for i, (ax, img) in enumerate(zip(axs.flatten(), compressed_images)):
            ax.imshow(img, cmap='gray')
            ax.set_title(f'{int((1 - compression_levels[i]) * 100)}% Compressed')
            ax.axis('off')
        plt.tight_layout()
        plt.show()
    
    def plot_runtime(self):
        sizes = [2**i for i in range(5, 11)]
        dft_times = []
        fft_times = []

        for size in sizes:
            signal = np.random.random(size)
            start = time.time()
            self.dft(signal)
            dft_times.append(time.time() - start)

            start = time.time()
            self.fft(signal)
            fft_times.append(time.time() - start)
        
        plt.figure(figsize=(10, 6))
        plt.plot(sizes, dft_times, label='Naive DFT', marker='o')
        plt.plot(sizes, fft_times, label='FFT', marker='o')
        plt.xlabel('Input Size')
        plt.ylabel('Time (seconds)')
        plt.title('Runtime Comparison: Naive DFT vs. FFT')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def validate_against_numpy(self):
        numpy_fft = np.fft.fft2(self.resized_image)
        custom_fft = self.fft_2d()
        difference = np.abs(numpy_fft - custom_fft)
        max_diff = np.max(difference)
        print(f"Max difference between numpy.fft2 and custom fft_2d: {max_diff}")

    def plot_denoise(self):
        frequency_domaine = self.fft_2d()

        magnitude = np.abs(frequency_domaine)
        frequency_domaine[magnitude > np.quantile(magnitude, 1 - 0.00099)] = 0

        denoised_image = self.ifft_2d(frequency_domaine)
        denoised_image = np.real(denoised_image)
        denoised_image = denoised_image[:self.original_image.shape[0], :self.original_image.shape[1]]

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(self.original_image, cmap='gray')
        axs[0].set_title("Original Image")

        axs[1].imshow(denoised_image, cmap='gray')
        axs[1].set_title("FFT Denoised Image")

        plt.show()
    
    def plot_fft(self):
        frequency_domaine = self.fft_2d()
        magnitude = np.abs(frequency_domaine)
        
        fft_image = magnitude[:self.original_image.shape[0], :self.original_image.shape[1]]


        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(self.original_image, cmap='gray')
        axs[0].set_title("Original Image")

        # Custom Fourier Transform (log scaled)
        axs[1].imshow(fft_image, norm=LogNorm(), cmap='gray')
        axs[1].set_title("FFT Image")
        
        plt.show()
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=int, required=False, help='Mode: 1 for Fast mode, 2 for Denoise, 3 for Compress, 4 for Plot runtime', default=1)
    parser.add_argument('-i', type=str, required=False, help='Filename of the image for the DFT', default='moonlanding.png')
    
    args = parser.parse_args()
    mode = args.m
    filename = args.i
    
    # Read the image in grayscale
    try:
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Cannot read the image file: {filename}")
    except Exception:
        raise ValueError(f"Cannot read the image file: {filename}")
    
    dft = DiscreteFourierTransform(image)
    
    # Perform requested mode
    if mode == 1:
        dft.plot_fft()
    elif mode == 2:
        dft.plot_denoise()
    elif mode == 3:
        dft.plot_compression()
    elif mode == 4:
        dft.plot_runtime()
    else:
        print('Invalid mode. Mode: 1 for Fast mode, 2 for Denoise, 3 for Compress, 4 for Plot runtime')
        exit()
    
if __name__ == '__main__':
    main()