1) Make sure you have a C compiler (GCC for linux or Visual Studio for windows should work)
2) Install the latest Cython package
3)If functions.pyx file does not exist, run the cython compiler notebook to generate one. If there wasn't a message indicating a successful compiliatiom, after making sure
the functions.pyx exists, run the following commands to generate the "functions" library and the cython report :

python setup.py build_ext --inplace
cython -a functions.pyx

4) If see an error message, either regarding Cython or the compiler, go back to steps 1 and 2

5) Run the "Layer generator examples.ipynb" notebook to create the appropriate coefficient and index layers. When you run the notebook, first choose the location and coefficent files for the retina. 
Then select the location and coefficent files for the left hemisphere of the cortex, then select the location and coefficent files for the right cortex hemisphere.

6) Run the "Retina examples.ipynb" notebook which contains 2 demos.