# spmsimu -- Simulator of Scanning Probe Microscopy

Scanning Probe Microscopy (SPM) simulator based on Python.

It simulates realistic SPM scans based on ground-truth patterns or user-input images. This simulator emulates effects of most of controlling parameters so it can be used as a training tool for new SPM operators.

It also serves as a playground for testing machine learning based automation algorithms as the results can be validated through real SPM experiments.


# 1. Installation

## 1-1. Quick installation
```Python
pip install spmsimu
```

To get started, please see the examples in the **"SpmSimu -- Tutorial 101.ipynb"** notebook located in [**spmsimu/notebooks**](https://github.com/RichardLiuCoding/spmsimu/tree/main/spmsimu/notebooks) folder.

# 2. Generate ground-truth patterns and tip shapes

In spmsimu package, we offer three ground-truth patterns for SPM scan simulation:

1. Checkerboard pattern
2. Spiral pattern
3. Atomic lattice with cubic structure
4. We also allow users to use their custom patterns or real topography maps as ground-truth patterns.

We also provide functions to generate single tip with different size and double tip with different separation, relative height, and sizes of each tip.

## Checkerboard pattern
```Python
checkerboard = generate_pattern(nx=256, ny=256, pattern='checkerboard', num=10, show=True)
```
![image](https://github.com/user-attachments/assets/68293f84-f5a8-48a7-b66d-7875258b2e42)

## Spiral pattern
```Python
spiral = generate_pattern(nx=256, ny=256, pattern='spiral', num=10, turns=5, show=True)
```
![image](https://github.com/user-attachments/assets/be0404ca-5876-48f4-8560-9856a9d3260f)

## Atomic lattice
```Python
atomic = generate_pattern(nx=256, ny=256, pattern='atomic', num=10, show=True)
```
![image](https://github.com/user-attachments/assets/fe4cdba2-9550-4840-afb3-6696a3b89951)

## Real topography maps
![image](https://github.com/user-attachments/assets/06e920ba-490e-42dd-8727-256704e4d7c7)
This one is included in the [**spmsimu/notebooks**](https://github.com/RichardLiuCoding/spmsimu/tree/main/spmsimu/notebooks) folder as an example.

## Ideal tip shape -- single sharp tip
```Python
tip_ideal = generate_tip_kernel(kernel_size=50, wx=5, wy=5)
```
![image](https://github.com/user-attachments/assets/3629a15e-3d33-4250-b0ed-6bdc3284bc6b)

## Double tip
```Python
tip_double = generate_doubletip_kernel(kernel_size=kernel_size, offset=offset,tip1=[wx1, wy1, amp1], tip2=[wx2, wy2, amp2])
```
![image](https://github.com/user-attachments/assets/5a505872-1768-4d47-87c3-8c6fb3ca55be)

# Simulate realistic scans

In the **scan()** function, there are following parameters can be tuned:
```Python
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
Output:
        Realistic scan image generated based on ground truth image, tip shape kernel, and scanning parameters.
```
## Scan with ideal parameters
```Python
traces = scan(image=checkerboard, kernel=tip_ideal, drive=0.5, setpoint=0.2,
          P=1, I=1e-2, z_speed=0.1, scan_speed=1, phase=True, retrace=True)
```
![image](https://github.com/user-attachments/assets/a8e7a63e-45ff-4a82-b49c-dc51d9b22bb2)

In an ideal scan, the trace and retrace match each other well:

![image](https://github.com/user-attachments/assets/790b0088-7d31-4f3d-8d3f-89f7e91a60cc)

## Scan with too slow z-speed or too fast scan speed -- parachutting effect
```Python
# Here we have decreased the z_speed from 1e-1 to 1e-2:
traces = scan(image=checkerboard, kernel=tip_ideal, drive=0.5, setpoint=0.2,
          P=1, I=1e-2, z_speed=1e-2, scan_speed=1, phase=True, retrace=True)
```
When the scan speed is too fast, the PI loop is not fast enough to respond to the fast change of sample height. As a result, we'll observe the parachutting effect:

![image](https://github.com/user-attachments/assets/a2eb4f67-5283-493b-97c7-69af2cdb6625)

Here the trace and retrace scans don't agree with each other:

![image](https://github.com/user-attachments/assets/d4bbfe56-4532-4567-8c50-2659955f3fce)

## Scan with too large I Gain -- unstable/oscillatory PI loop
```Python
# Here we have decreased the I Gain from 1e-2 to 1:
traces = scan(image=checkerboard, kernel=tip_ideal, drive=0.5, setpoint=0.2,
          P=1, I=1, z_speed=1e-1, scan_speed=1, phase=True, retrace=True)
```
![image](https://github.com/user-attachments/assets/ce6201ce-2da1-4cb2-b0bd-f846214a2ca3)

![image](https://github.com/user-attachments/assets/46e2d787-df99-4192-9a89-391e67d00919)

# Other uses of the package
## Double tip effect

![image](https://github.com/user-attachments/assets/9ffdd2d0-ad3d-4aba-9cad-ea25e9c6a32b)

## Tip change event
![image](https://github.com/user-attachments/assets/08dca27b-6f69-419e-b18a-628c3844cdc6)





