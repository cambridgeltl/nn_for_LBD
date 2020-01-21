# ReadMe

This repo contains the files that implemented the experiments and the data used in the paper 'Neural networks for open and closed literature based discovery' by Gamal Crichton, Simon Baker, Yufan Guo and Anna Korhonen.

The shell file contains the steps to automate the experiments. The .py files are used in the batch script.

The LINE folder contains implementations of the LINE algorithm obtained freely on the internet.

The *Models* folder contains the files for the models used. 

The data files used were too large for Github's file size limit. The Cancer landmark discovery and Swanson discovery datasets are the same as those used in the LION LBD paper and can be found at: http://lbd.lionproject.net/downloads. The BioGRID dataset used is from https://downloads.thebiogrid.org/BioGRID. Since the BioGRID dataset updates frequently, the exact dataset we used (version 3.4.167) would likely be archived; so can be found at: https://downloads.thebiogrid.org/BioGRID/Release-Archive/.

The parameters that each script accepts are in the files.

Feel free to open an issue if anything does not work.

**Dependencies**
Dependencies are in the requirements.txt file.

Major ones:

+ Python 2.7
+ Pytorch
+ Networkx
+ SciKit-learn
+ Numpy
+ Pandas

## License
The code is provided under MIT license and the other materials under Creative Commons Attribution 4.0.
