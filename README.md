# GemStone Gem5-Validate 

This project analyses gem5 simulation data, combines it with data collected 
from a hardware platform (e.g. collected with 
[GemStone-Profiler-Automate](https://github.com/mattw200/gemstone-profiler-automate)) and applies
 statistical techniques to observe the difference between the gem5 modle and 
 real hardware platform, and identify sources of errors in the model. 
 
It has three main functions handled by three main scripts:

1. `gem5_combine_experiments.py`:  Automatically finds gem5 experiments, converts the stats.txt file 
		into a csv file and then combines the experiments into one tab-separated 
		csv file. A tutorial on this step is actually presented at the end of the 
		gem5 and GemStone-Gem5-Automate tutorial.
2. `combine_gem5_hw.py`: Combines the processed data from the hardware platform with the gem5 
		experiment data (described in this tutorial).
3. `validation.py`: Conducts the correlation, hierarchical cluster, and regression analysis 
		as well as direct hardware PMC event to modelled gem5 event comparisons.
		
## Usage Instructions

Usage instructions and tutorials available at [GemStone](http://gemstone.ecs.soton.ac.uk). 


## Authors

[Matthew J. Walker](mailto:mw9g09@ecs.soton.ac.uk) - [University of Southampton](https://www.southampton.ac.uk)

This project supports the paper:
>M. J. Walker, S. Bischoff, S. Diestelhorst, G V. Merrett, and B M. Al-Hashimi,
>["Hardware-Validated CPU Performance and Energy Modelling"](http://www.ispass.org/ispass2018/),
>in IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS), 
> Belfast, Northern Ireland, UK, April, 2018 [Accepted]

This work is supported by [Arm Research](https://developer.arm.com/research), 
[EPSRC](https://www.epsrc.ac.uk), and the [PRiME Project](http://www.prime-project.org).


## License

This project is licensed under the 3-clause BSD license. See LICENSE.md for details.


