# SARA_ScienceAdvances

The raw data is available in ./Bi2O3. To process the data follow the instructions below.

## Extract features from images
Execute
```
python get_gp-bias.py
```
This script will extract the features from the images and create separate plots, which contain the image itself as well as the RGB and LSA bias.
The bias features will be written to a file called `bias.json`

## Process and plot reflectance spectroscopy
Execute
```
python get_legcoeff.py
```
This script will read and normalize the reflectance data, and expand the spectra in Legendre coefficients.
The coefficients will be written to a file called `legendre_coefficients.json`

## References
Software written by Maximilian Amsler, released on 7/24/2021


If using this work for a publication, please cite:
"Autonomous synthesis of metastable materials", 
Sebastian Ament, Maximilian Amsler, Duncan R. Sutherland, Ming-Chiang Chang, Dan Guevarra, Aine B. Connolly, John M. Gregoire, Michael O. Thompson, Carla P. Gomes, R. Bruce van Dover, arXiv:2101.07385 

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
