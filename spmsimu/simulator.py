import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numba import jit, njit, prange
import random


###------Here are functions to generate ground-truth patterns------###

norm_ = lambda x: (x - np.min(x)) / np.ptp(x) * 1

def normalize(x, center=False):
    '''
    Normalize the data to [0, 1], with the option to shift it to [-0.5, 0.5].
    Inputs:
        x:  -ndarry: 1D or 2D image array to be normalized.
        center:     -boolean: controls if the output is shifted to be centered at 0.
    Outputs:
        x_norm -ndarray: normalzied image array
    '''
    if center:
        return (x - np.mean(x)) / np.ptp(x)
    else:
        return (x - np.min(x)) / np.ptp(x)

def generate_pattern(nx, ny, pattern='checkerboard', num=10, turns=3, show=False):
    '''
    Generate ground-trutn patterns for scan simulation.
    
    Inputs:
        nx - int: number of pixels along x direction
        ny - int: number of pixels along y direction
        pattern - string: choose between "checkerboard", "spiral", and "atomic"
        num - int: number of features (boxes, spirals, or atoms) in the pattern
        turns - int: controls the curvature of spirals (only for "spiral")
        show - boolean: if True, the generated pattern will be displayed
        
    Output:
        pattern_map - ndarray: 2D map with ground-truth patterns
    '''
    output = np.zeros((nx, ny))
    box_size = nx // num
    center_x, center_y = nx // 2, ny // 2
    
    if pattern == 'checkerboard':
        for i in range(nx):
            for j in range(ny):
                if (i + j) % 2 == 0:
                    output[i * box_size:(i + 1) * box_size,
                           j * box_size:(j + 1) * box_size] = 1  # Set to white
    elif pattern == 'spiral':
        for x in range(nx):
            for y in range(ny):
                # Calculate the distance and angle from the center
                dx, dy = x - center_x, y - center_y
                distance = np.sqrt(dx**2 + dy**2)
                angle = np.arctan2(dy, dx)

                # Generate a continuous spiral effect
                # Adjust frequency by `turns`, which determines the number of spiral turns
                spiral_value = 0.5 * (1 + np.sin(distance / box_size + turns * angle))
                # spiral_value = 1
                output[y, x] = spiral_value  # Assign grayscale value between 0 and 1
            output[np.where(output>0.5)] = 1
            output[np.where(output<=0.5)] = 0
    elif pattern == 'atomic':
        kx = 2*np.pi/nx * num
        ky = 2*np.pi/ny * num
        x = np.arange(nx)
        y = np.arange(ny)
        X, Y = np.meshgrid(x,y)
        output = np.cos(kx*X)*np.cos(ky*Y) / 2 + 0.5
    else:
        print("Error: please choose pattern between 'checkerboard', 'spiral', and 'atomic'.")
    
    if show:
        fig,ax=plt.subplots(1,2,figsize=[9,4])
        ax[0].imshow(output)
        ax[0].axhline(nx//2, linestyle='--', color='red')
        ax[0].axis('off')
        ax[1].plot(output[nx//2,:])
        plt.tight_layout()
    
    return output
    
###------Here are functions to simulate tip effects------###

@jit(nopython=True, parallel=True)
def pad_image(image, pad_height, pad_width):
    '''
    Pad the image with -1 on the four edges to make sure we can run the kernel through all the pixels.
    Inputs:
        image:    -ndarray: 2D image array to be simulated based on
        pad_height -int: kernel_height // 2
        pad_width  -int:  kernel_width // 2
    Outputs:
        padded_image -ndarray: 2D image array with edge extented by padding -1
    '''
    image_height, image_width = image.shape
    padded_height = image_height + 2 * pad_height
    padded_width = image_width + 2 * pad_width
    padded_image = -np.ones((padded_height, padded_width))  # Use constant value -1 for padding

    for i in prange(image_height):
        for j in prange(image_width):
            padded_image[i + pad_height, j + pad_width] = image[i, j]

    return padded_image


def generate_tip_kernel(kernel_size=50, wx=5, wy=5, tip_height=1):
    # Kernel size in pixels
    nx, ny = kernel_size, kernel_size

    center1 = np.array([nx//2, ny//2])

    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)

    return np.exp(-((X-center1[0])**2/(2*wx**2) + (Y-center1[1])**2/(2*wy**2))) * tip_height

def generate_doubletip_kernel(kernel_size=50, offset=[1,0], tip1=[5,5,1], tip2=[5,9,0.5]):
    '''
    Generate the kernel for double tip, with tip1: [x_width, y_width, tip_height]
    The two tips are separated by offset.
    Everything is in the unit of pixels.
    '''
    # Kernel size in pixels
    nx, ny = kernel_size, kernel_size

    wx1, wy1, amp1 = tip1
    wx2, wy2, amp2 = tip2
    center1 = np.array([nx//2, ny//2])
    center2 = np.array([nx//2, ny//2]) + np.array(offset)

    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)

    return np.exp(-((X-center1[0])**2/(2*wx1**2) + (Y-center1[1])**2/(2*wy1**2))) * amp1 + \
            np.exp(-((X-center2[0])**2/(2*wx2**2) + (Y-center2[1])**2/(2*wy2**2))) * amp2
            

@jit(nopython=True, parallel=True)
def tip_scan(image, kernel):
    '''
    Scanning image simulated with real probe shapes defined by the kernel.
    '''
    # image = norm_(image)
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    padded_image = pad_image(image, pad_height, pad_width)

    output = np.zeros((image_height, image_width))

    for i in prange(image_height):
        for j in prange(image_width):
            crop = padded_image[i:i + kernel_height, j:j + kernel_width]
            output[i, j] = 1 - np.min(2 - kernel - crop)

    return output

###------Here are functions to simulate PI loops------###

class FixedLengthStack:
    '''
    Numpy circular stack with adjustable length to store
    '''
    def __init__(self, length=10):
        self.buffer = np.zeros(length, dtype=np.float64)  # Initialize with zeros or any default value
        self.length = length
        self.index = 0
        self.full = False

    def append(self, value):
        self.buffer[self.index] = value
        self.index = (self.index + 1) % self.length
        if self.index == 0:
            self.full = True

    def get_stack(self):
        if not self.full:
            return self.buffer[:self.index]
        return np.concatenate((self.buffer[self.index:], self.buffer[:self.index]))

    def sum_stack(self):
        if not self.full:
            return np.sum(self.buffer[:self.index])
        return np.sum(self.buffer)

# @title
def pi_loop(trace, setpoint=0.1, P=1e0, I=1e-2, z_0=0.1, length=10):
    output = np.zeros_like(trace)

    if len(np.shape(trace)) == 1:
        output[0] = trace[0] + setpoint

        integral = FixedLengthStack(length=length)
        for i in range(len(trace)-1):
            z_diff = output[i]-trace[i]-z_0
            integral.append(z_diff)
            output[i+1] = output[i] - P*(z_diff) - I*integral.sum_stack()
        return output
    elif len(np.shape(trace)) == 2:
        for index in range(len(trace)):
            output[index, 0] = trace[index, 0] + setpoint
            integral = FixedLengthStack(length=length)
            for i in range(len(trace[index])-1):
                z_diff = output[index, i]-trace[index, i]-z_0
                integral.append(z_diff)
                output[index, i+1] = output[index, i] - P*(z_diff) - I*integral.sum_stack()
        return output

###------Here are functions to simulate tip changes------###

def scanning_tip_change(img, kernel1, pi1, kernel2=None, pi2=None):
    '''
    Simulate the scanning with tip change and/or PI setting changes.
    Input:
        img: ground truth image pattern
        kernel1: tip kernel before tip change event
        kernel2: tip kernel after tip change event. If not provided, will be the same as kernel1
        pi1: [P, I] settings for the PI before tip change event
        pi2: [P, I] settings for the PI after tip change event. If not provided, will be the same as pi1
    '''
    if kernel2 is None:
        kernel2 = kernel1
    if pi2 is None:
        pi2 = pi1
    out1 = pi_loop(tip_scan(img, kernel1), P=pi1[0], I=pi1[1])
    out2 = pi_loop(tip_scan(img, kernel2), P=pi2[0], I=pi2[1])

    nx, ny = np.shape(out1)

    index = random.randint(0, nx * ny)

    out1 = out1.flatten()
    out1[index:] = out2.flatten()[index:]

    return out1.reshape([nx, ny])
    
###------Here are functions to include everything together------###
    
@njit
def generate_phase(x, setpoint=0.5):
    return np.where(
        x < np.sqrt(setpoint)/1.5 + 0.3,
        -np.arccos(x) * 50 + 90,
        np.where(x < 1, np.arccos(x) * 50 + 90, 90)
        )

    # return np.where(
    #     x < setpoint,
    #     -np.arccos(x) * 50 + 90,
    #     np.where(x < 1, np.arccos(x) * 50 + 90, 90)
    # )

    
@njit
def generate_random_walk_2D(num_traces, N, step_size=0.1):
    '''
    Generate a 2D array of random walk traces.
    
    Input:
        num_traces - int: the number of random walk traces to generate
        N          - int: the length of each random walk trace
        step_size  - float: the maximum step size (uniformly sampled between -step_size and step_size)

    Output:
        ndarray: a 2D array of shape (num_traces, N + 1), where each row is a random walk trace
    '''
    # Initialize the output array with zeros
    positions = np.zeros((num_traces, N + 1))
    
    # Loop through each trace
    for i in range(num_traces):
        # Generate random steps
        steps = np.random.uniform(-step_size, step_size, N)
        # Compute cumulative sum manually
        for j in range(N):
            positions[i, j + 1] = positions[i, j] + steps[j]
    
    return positions[:,1:]

@njit(parallel=True)
def scan(image, kernel, drive=0.5, setpoint=0.2, P=1e0, I=1e-2, length=10, z_speed=0.1, scan_speed=1,
         phase=False, retrace=False, noise=False):
    '''
    Generate realistic scan images based on the ground truth image, tip shape kernel, and scanning parameters.
    Input:
        image      - ndarray: ground truth 1D or 2D image profile
        kernel     - ndarray: tip shape kernel
        drive      - float: the drive amplitude (free-air amplitude)
        setpoint   - float: setpoint amplitude (setpoint tip-sample distance in the simulator)
        P          - float: proportional gain in PID
        I          - float: integral gain in PID
        length     - int: the number of pixels in "memory" of PID algorithm
        z_speed    - float: the extend/retrace speed of the z piezo
        scan_speed - float: the xy movement speed of the tip
        phase      - boolean: if true, a corresponding phase map is generated along with the height map
        retrace    - boolean: if true, both trace and retrace maps will be generated
        noise      - boolean: if true, when the drive is too small and setpoint is too large, scans will be dominated by noise

    output:
        Realistic scan image generated based on ground truth image, tip shape kernel, and scanning parameters.
    '''
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Output always has four channels -- z, phase, z_re, phase_re
    output = np.zeros((4, image_height, image_width))

    if noise and (setpoint > (np.sqrt(drive))/5+0.8):
        h0 = setpoint
        ph0 = 95
        output[0] = h0 + generate_random_walk_2D(image_height, image_width, step_size=0.1)
        output[2] = h0 + generate_random_walk_2D(image_height, image_width, step_size=0.1)
        output[1] = ph0 + generate_random_walk_2D(image_height, image_width, step_size=0.1)
        output[3] = ph0 + generate_random_walk_2D(image_height, image_width, step_size=0.1)
        return output
    else:
        pad_height = kernel_height // 2
        pad_width = kernel_width // 2

        # Pad the boundaries with -1
        padded_image = pad_image(image, pad_height, pad_width)

        z_groundtruth = np.zeros((image_height, image_width))

        # Generate the ground truth map with tip kernels
        for i in prange(image_height):
            for j in prange(image_width):
                crop = padded_image[i:i + kernel_height, j:j + kernel_width]
                z_groundtruth[i, j] = 1 - np.min(2 - kernel - crop)

        z_cal = np.zeros_like(image)
        z_measured = np.zeros_like(image)
        delta_z_max = z_speed / scan_speed

        # Initialize placeholder arrays
        z_measured_re = np.zeros_like(image) if retrace else np.zeros((0, 0))
        phase_map = np.zeros_like(image) if phase else np.zeros((0, 0))
        phase_map_re = np.zeros_like(image) if (phase and retrace) else np.zeros((0, 0))

        # Initialize PID integral stack per scan line to avoid shared memory issues
        # Here we take the buffer out of for loop to keep a memory on the previous scan lines
        # integral = np.zeros(length)
        
        # for index in prange(len(z_groundtruth)):  # Parallel over scan lines
        for index in prange(len(z_groundtruth)):  # Parallel over scan lines
            
            # Start scanning from the beginning of each line
            z_cal[index, 0] = z_groundtruth[index, 0] + setpoint
            z_measured[index, 0] = z_groundtruth[index, 0] + setpoint

            integral = np.zeros(length)
            
            for i in range(len(z_groundtruth[index])-1):
                z_diff_cal = z_measured[index, i] - z_groundtruth[index, i] - setpoint

                # Update integral (shift left and add new value)
                integral = np.roll(integral, -1)
                integral[-1] = z_diff_cal

                integral_sum = np.sum(integral)
                distance_to_move = P * z_diff_cal + I * integral_sum

                if np.abs(distance_to_move) > delta_z_max:
                    distance_to_move = np.sign(distance_to_move) * delta_z_max

                z_measured[index, i+1] = z_measured[index, i] - distance_to_move

        # Generate retrace map if requested
        if retrace:
            # integral = np.zeros(length)  
            # for index in prange(len(z_groundtruth)):  # Parallel over scan lines for retrace
            for index in prange(len(z_groundtruth)):  # Parallel over scan lines for retrace
                # Initialize integral stack per scan line
                integral = np.zeros(length)

                z_measured_re[index, 0] = z_groundtruth[index, -1] + setpoint

                for i in range(len(z_groundtruth[index])-1):
                    z_diff_cal_re = z_measured_re[index, i] - z_groundtruth[index][::-1][i] - setpoint

                    # Update integral (shift left and add new value)
                    integral = np.roll(integral, -1)
                    integral[-1] = z_diff_cal_re

                    integral_sum = np.sum(integral)
                    distance_to_move = P * z_diff_cal_re + I * integral_sum

                    if np.abs(distance_to_move) > delta_z_max:
                        distance_to_move = np.sign(distance_to_move) * delta_z_max

                    z_measured_re[index, i+1] = z_measured_re[index, i] - distance_to_move
                    
            output[2] = z_measured_re
            
        output[0] = z_measured
        
        # Generate phase map if requested
        if phase:
            phase_map = generate_phase(z_measured - z_groundtruth, drive)
            output[1] = phase_map
            if retrace:
                phase_map_re = generate_phase(z_measured_re[:, ::-1] - z_groundtruth, drive)
                output[3] = phase_map_re
        
        return output