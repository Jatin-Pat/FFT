import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cv2
import argparse

class DiscreteFourierTransform:
    def __init__(self, image):
        self.image = image
        
        
    def dft(self, signal):
        N = len(signal)
        X = np.zeros(N, dtype=complex)
        
        for k in range(N):
            for n in range(N):
                X[k] += signal[n] * np.exp(-2j * np.pi * k * n / N)
                
        return X
    
    def fft(self, signal, size_threshold=8):
        N = len(signal)
        
        if N <= size_threshold:
            return self.dft(signal)
        
        even = self.fft(signal[0::2])
        odd = self.fft(signal[1::2])
        
        coeff = np.exp(-2j * np.pi * np.arange(N) / N)
        
        return np.concatenate([even + coeff[:N//2] * odd, even + coeff[N//2:] * odd])
    
    def fft_2d(self):
        row_transformed_image = np.zeros_like(self.image, dtype=complex)
        for i in range(self.image.shape[0]):
            row_transformed_image[i, :] = self.fft(self.image[i, :])
            
        col_transformed_image = np.zeros_like(row_transformed_image, dtype=complex)
        for j in range(row_transformed_image.shape[1]):
            col_transformed_image[:, j] = self.fft(row_transformed_image[:, j])
        
        return col_transformed_image
    
    def display_fft(self):
        fft_result = self.fft_2d()

        # Take the magnitude of the FFT result for visualization
        magnitude_spectrum = np.abs(fft_result)

        # Plot the original and transformed images
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Original image
        axs[0].imshow(self.image, cmap='gray')
        axs[0].set_title("Original Image")
        axs[0].axis('off')

        # Fourier Transform (log scaled)
        axs[1].imshow(magnitude_spectrum, norm=LogNorm(), cmap='gray')
        axs[1].set_title("Fourier Transform (Log Scale)")
        axs[1].axis('off')

        plt.show()
        
                
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=int, required=False, help='Mode: 1 for Fast mode, 2 for Denoise, 3 for Compress, 4 for Plot runtime', default=1)
    parser.add_argument('-i', type=str, required=False, help='Filename of the image for the DFT', default='moonlanding.png')
    
    args = parser.parse_args()
    mode = args.m
    filename = args.i
    
    # Read the image
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    
    #Create DFT class
    dft = DiscreteFourierTransform(image)
    
    # Perform requested mode
    if mode == 1:
        dft.display_fft()
    elif mode == 2:
        dft.denoise()
    elif mode == 3:
        dft.compress()
    elif mode == 4:
        dft.plot_runtime()
    else:
        print('Invalid mode. Mode: 1 for Fast mode, 2 for Denoise, 3 for Compress, 4 for Plot runtime')
        exit()
    
    return 0
    
if __name__ == '__main__':
    main()