# A Journey to the center of the 3D Space

## Testing Environment
1. We tested the hardware code generation on three different machines based on Ubuntu 18.04, 20.04 and Cent OS 7 respectively.
2. We used Xilinx Vitis Unified Platform and Vivado HLx toolchains 2019.2
3. We used python 3.6 with `argparse` `numpy` `math` 'pandas' packets on the generation machine
4. a) As host machines, or hardware design machines, we used Pynq 2.7 on the Zynq based platforms (Zcu104, Alveo u200), where we employ `cv2`, `numpy`, `pandas`, `multiprocessing`, `statistics`, `argparse`, and `pydicom` packetes.
4. b) We tested the Alveo u200 on a machine with CentOS 7.6, i7-4770 CPU @ 3.40GHz, and 16 GB of RAM, and we installed Pynq 2.7 following the [instructions by the Pynq team](https://pynq.readthedocs.io/en/v2.5.1/getting_started/alveo_getting_started.html) with the same packets as point 4a.
5. [Optional] Possible issues with locale: export LANG="en_US.utf8"

## Code organization
* `src/` source code for HLS based design, miscellaneous utilities, python host and testing code, and various scripts
 * `hls/` HLS source code for both design and testbench
 * `scripts/` miscellaneous scripts for the design generation, from tcl for Vivado and Vivado HLS to design configurator and design results extractions, as well as for python host source code for single MI and multiple MI tests
* `platforms/` specific platforms makefile for the current supported boards: Pynq-Z2, Ultra96, Zcu104, Alveo u200


## FPGA-based Mutual Information (MI) accelerator generation flow

1. Source the necessary scripts, for example: `source <my_path_to_vitis>/settings64.sh`; for Alveo you will need to source xrt, e.g., `source /opt/xilinx/xrt/setup.sh`
2. Just do a `make`, or `make help` in the top folder for viewing an helper (print all helpers  `make helpall` )
3. use/modify the design space exploration script (i.e., `dse.sh` or `top_build.sh`) or generate your designs or use single instance specific generation 
4. a) `make hw_gen TRGT_PLATFORM=<trgt_zynq>` for generating an instance of a Zynq-based design, where `trgt_zynq=zcu104|ultra96|pynqz2`
4. b) `make hw_gen TARGET=hw OPT_LVL=3 CLK_FRQ=$FREQZ TRGT_PLATFORM=alveo_u200 NCM=$NUMCOUPLES` for generating an instance of the design on the Alveo u200 with target clock frequency `CLK_FRQ=$FREQZ`and $NUMCOUPLES as maximum number of supported couples
5. [Optional] Generate other instances changing the design parameters. Look at Makefile parameters section for details.

## Testing designs

1. Complete at least one design in the previous section
2. `make sw` creates a deploy folder for the python code
3. `make deploy BRD_IP=<target_ip> BRD_USR=<user_name_on_remote_host> BRD_DIR=<path_to_copy>` copy onto the deploy folders the needed files
4. connect to the remote device, i.e., via ssh `ssh <user_name_on_remote_host>@<target_ip>`
5. [Optional] install all needed python packages as above, or the pynq package on the Alveo host machine
6. set `BITSTREAM=<path_to_bits>`, `CLK=200`, `CORE_NR=<target_core_numbers>`, `PLATFORM=Alveo|Zynq`, `RES_PATH=path_results`, and source xrt on the Alveo host machine,  e.g., `source /opt/xilinx/xrt/setup.sh`

Three possible tests can be executed. The first one executes, from CLI, tests with increasing number of couples, from 1 up to a given parameter, and outputs, to a csv file, data related to the output MI, the mean and standard deviation of hardware execution time, the mean and standard deviation of error (hardware vs software) and the mean and standard deviation of software time, by testing with different seeds for random values generation. To execute it, please execute the following command onto the host of the board:

python3 3DImageRegistration_custom.py -ol <bitstream filename> -nc <max_ncouples> -f <output_filename> -im <image_dimension>
Other parameters can be inserted to.

The second type of test is a python script in which the source can be modified to manually set parameters. To do so, modify the source code of 3DImageRegistration.py, by modifying, in the main, "platform" (to 'Alveo' or 'Zynq', according to the board used), and "overlay" to the path of the bitstream. Other parameters can be modified too. The output file gives the same content as the first type of test.

The third type of test is a Jupyter Notebook: 3D Image Registration.ipynb, which shows a single computation of the output MI, for given seed and n_couples.
Please notice that if on Zynq you will need 'sudo'.

### Makefile parameters

Follows some makefile parameters

#### General makefile parameters, and design configuration parameter
* TRGT_PLATFORM=`pynqz2|ultra96_v2|zcu104|alveo_u200`
* Histogram Computation type HT=`float|fixed`
* Histogram PE Number PE=`1|2|4|8|16|32|64` 
* Entropy PE Number PE_ENTROP=`1|2|4|8|16|32`
* Use caching or not CACHING=`true|false`
* Use URAM caching URAM=`true|false`, not effective if CACHING=`false`
* Core Number CORE_NR=`1|2|3|4`
* Maximum number of supported couples NCM

#### Vivado and Zynq specific parameters flow
* HLS_CLK=`default 10` clock period for hls synthesis
* FREQ_MHZ=`150` clock frequency for vivado block design and bitstream generation
* TOP_LVL_FN=`mutual_information_master` target top function for HLS
* HLS_OPTS=`5` HLS project flow. Supported options: 0 for only project build; 1 for sim only; 2 synth; 3 cosim; 4 synth and ip downto impl; 5 synth and ip export; 6 for ip export

#### Alveo specific parameters flow
* REPORT_FLAG=`R0|R1|R2` to report detail levels
* OPT_LVL=`0|1|2|3|s|quick` to optimization levels
* CLK_FRQ=`<target_mhz>` to ClockID 0 (board) target frequency, should be PCIe clock

## Extracting resources results

1. a) `make resyn_extr_zynq_<trgt_zynq>` e.g., trgt_zynq=`zcu104|ultra96|pynqz2`, set FREQ_MHZ parameter if different from default.
1. b) `make resyn_extr_vts_<trgt_alveo>` e.g., trgt_alveo=`alveo_u200`
2. You will find in the `build/` folder a new folder with all the generated bitstreams, and in the `build/<TRGT_PLATFORM>/` directory you will find a .csv with all the synthesis results


#### Credits and Contributors

Contributors: Giuseppe Sorrentino, Marco Venere
Based on IRON: A Framework for Customizable FPGA-based Image Registration Accelerators
Credits to: Conficconi, Davide and D'Arnese, Eleonora and Del Sozzo, Emanuele and Sciuto, Donatella and Santambrogio, Marco D.

If you find this repository useful, please use the following citation(s):

```
@inproceedings{iron2021,
author = {Conficconi, Davide and D'Arnese, Eleonora and Del Sozzo, Emanuele and Sciuto, Donatella and Santambrogio, Marco D},
title = {A Framework for Customizable FPGA-based Image Registration Accelerators},
booktitle = {The 2021 ACM/SIGDA International Symposium on Field-Programmable Gate Arrays},
pages={251--261},
year = {2021}
}
```