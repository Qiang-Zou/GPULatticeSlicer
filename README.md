# GPU Slicer for Lattice Structures (GPULatticeSlicer)

- By: Qiang Zou and Sifan Chen
-  email: qzou.code@gmail.com
- webpage: https://qiang-zou.github.io/
- Latest Release: 2022.12

## !important
The source code is a GPU-based parallel implicit slicer for modeling and fabricating highly complex lattice structures, built on top of the idea presented in [1,2]. It can be compiled with QT 5.10+MSVC 14.0, and run on the operating system Windows 10.

1.Copyright
-----------

- GPULatticeSlicer is developed by Qiang Zou and Sifan Chen for research use only. All rights about the program are reserved by Qiang Zou. This C++ source codes are available only to a primary user for academic purposes. No secondary use, such as copy, distribution, diversion, business purpose, etc., is allowed. In no event shall the author be liable to any party for direct, indirect, special, incidental, or consequential damage arising out of the use of this program. ImplicitSlicer is self-contained. 


2.Download
----------

- The source code, as well as the testing data, can be downloaded from the page: 
  
  webpage: https://github.com/Qiang-Zou/GPULatticeSlicer/


3.Installing & Compiling (Windows+QT5.10+MSVS14.0+CUDA10.2)
-------------------------------------------

- Simply download the source code to a suitable place, add Eigen Library to the root directory, and use QT5.10+MSVC14.0 to build the project.
- Also, almost all codes are Cuda-based, so please make sure you have set up this library before using this code.

4.Usage
-------

- After the compilation you can run the tool 3DPrintPlanner.exe inside the ./build/release/ directory:

	- **Note** It is recommended to use the "Run in terminal" mode, not mandatory though.  
	- **Note** The data files should be located in ./Data directory. Before using the tool, please unpack all model files to the directory ./Data in advance.  

- To slice a model, you can simply right click in the GUI, and navigate to "Mesh -> Lattice Modeling". After that you will be asked to input several necessary parameters like layer height through the terminal to run the program, and then the slicing will start working.

- For each layer, the contours are stored using the variable "result". You can add your customized processing code in the following section:

		// LatticeModeler.cpp
		line 399   auto result = msb.getContours(false);
		line 340   //.......................
			// do things with "result"
			
		line 343  //........................



5.File format
-------------

- Edge file: lattice edge file (placed in ./Data, and any extension can be used).

	- line 1:	edge_number edge_radius convoluation_radius convoluation_kernel_size
	- lines 2-n:	x1 y1 z1 x2 y2 z2
	- **Note** x1 y1 z1: coordinates of one end point; x2 y2 z2: coordinates of the other end point


6.References
-------------
- [1] Shengjun Liu, Tao Liu, Qiang Zou, Weiming Wang, Eugeni L. Doubrovski, and Charlie C.L. Wang, "Memory-efficient modeling and slicing of large-scale adaptive lattice structures", ASME Transactions - Journal of Computing and Information Science in Engineering, Accepted.
- [2] Junhao Ding, Qiang Zou, Shuo Qu, Paulo Bartolo, Xu Song, Charlie C. L. Wang, A new digital design and manufacturing paradigm for high precision powder bed fusion process. Under Review.

