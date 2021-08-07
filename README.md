# SARA_ScienceAdvances

The raw data is available from eCommons (link). Download the image/spectroscopy data separately, and to process the data follow the instructions below.

## Preprocessing
Before running below scripts, copy the raw data from microscopy imaging and reflectance measurements available from <https://doi.org/10.7298/h63q-9r54> to `ProcessingData/Bi2O3`, or modify the paths in the relevant python files.

## Extract features from images
In `ProcessingData`, execute
```
python get_gp-bias.py
```
This script will extract the features from the images and create separate plots, which contain the image itself as well as the RGB and LSA bias.
The bias features will be written to a file called `bias.json`.

## Process and plot reflectance spectroscopy
In `ProcessingData`, execute
```
python get_legcoeff.py
```
This script will read and normalize the reflectance data, and expand the spectra in Legendre coefficients.
The coefficients will be written to a file called `legendre_coefficients.json`.

## Active learning set up

Process the `legendre_coefficients.json` and `bias.json` files using the `inner_data_organizer.jl` script in `SARA.jl/ScienceAdvances2021/inner/`. Make sure to adjust the path variable in the script to match the location of the `.json` files. 
Using a version of Julia greater than 1.6.2, execute the `install.jl` file in `GaussianDistributions.jl` and `SARA.jl`, in that order. 
If the add operation of `GaussianDistributions.jl` for `SARA.jl` fails, add the package locally by typing `]add path/to/GaussianDistributions.jl\` in the REPL.
After that, everything is set up for the active learning benchmarks.

## Active learning benchmarks for characterization loop

In the `SARA.jl/ScienceAdvances2021/inner/` directory, there are the `inner_kernel_benchmark.jl` and `inner_acquisition_benchmark.jl` files which can be used to reproduce the results reported in Figure 2 of the main article. 
`inner_kernel_plot.jl` and `inner_acquisition_plot.jl` can further create the plots used in the figure.

## Active learning benchmarks for synthesis loop

In the `SARA.jl/ScienceAdvances2021/outer/` directory, there are the `outer_kernel_benchmark.jl` and `outer_acquisition_benchmark.jl` files which can be used to reproduce benchmark results for the synthesis loop, the acquisition benchmark being reported in Figure 3 of the main article. 
`outer_kernel_plot.jl` and `outer_acquisition_plot.jl` can further create the plots for the benchmarks.
`outer_gradient_map.jl` records the gradient maps for varying kernel lengthscale parameters and stripe sampling techniques, used to explore the behavior of the results with respect to changes in these hyper-parameters.
`outer_gradient_learning.jl` executes active learning and records the evolution of the gradient maps and `outer_gradient_plot.jl` plots the corresponding results as shown in Figure 4 of the main article.



## References
Software written by Maximilian Amsler and Sebastian Ament, released on 7/24/2021.


If using this work for a publication, please cite:
"Autonomous synthesis of metastable materials", 
Sebastian Ament, Maximilian Amsler, Duncan R. Sutherland, Ming-Chiang Chang, Dan Guevarra, Aine B. Connolly, John M. Gregoire, Michael O. Thompson, Carla P. Gomes, R. Bruce van Dover, [arXiv:2101.07385](https://arxiv.org/abs/2101.07385) 

## Disclaimer

THE SOFTWARE IS PROVIDED 'AS IS' WITHOUT ANY WARRANTY OF ANY KIND, EITHER
EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY
WARRANTY THAT THE SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
FREEDOM FROM INFRINGEMENT, AND ANY WARRANTY THAT THE DOCUMENTATION WILL
CONFORM TO THE SOFTWARE, OR ANY WARRANTY THAT THE SOFTWARE WILL BE ERROR
FREE. IN NO EVENT SHALL THE DEVELOPER BE LIABLE FOR ANY DAMAGES, INCLUDING, BUT
NOT LIMITED TO, DIRECT, INDIRECT, SPECIAL OR CONSEQUENTIAL DAMAGES,
ARISING OUT OF, RESULTING FROM, OR IN ANY WAY CONNECTED WITH THIS
SOFTWARE, WHETHER OR NOT BASED UPON WARRANTY, CONTRACT, TORT, OR
OTHERWISE, WHETHER OR NOT INJURY WAS SUSTAINED BY PERSONS OR PROPERTY OR
OTHERWISE, AND WHETHER OR NOT LOSS WAS SUSTAINED FROM, OR AROSE OUT OF
THE RESULTS OF, OR USE OF, THE SOFTWARE OR SERVICES PROVIDED HEREUNDER.
