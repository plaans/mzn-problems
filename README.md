# mzn-problems

mzn-problems is a collection of MiniZinc problems and a script to test aries solver. Most, if not all, the problems come from Hakan Kjellerstrand's website https://www.hakank.org/.

Titouan Seraud - [titouan.seraud@laas.fr](mailto:titouan.seraud\@laas.fr) <!-- titouan.seraud@insa-toulouse.fr -->

<details>
<summary><b>Table of contents</b></summary>

- [Installation](#installation)
- [Usage](#usage)
- [Documentation](#documentation)
- [Versioning](#versioning)
- [Useful links](#useful-links)
</details>


## Installation

### Minizinc
Make sure you have installed [minizinc](https://www.minizinc.org/). Verify the solver you want to test is correctly referenced in `MZN_SOLVER_PATH`. If it is not present, you can add it using the following command.
```bash
export MZN_SOLVER_PATH=/path/to/my/solver/share
```

### Python
The python script `mznb.py` does not use any external depedency. Hence, it should be very easy to use. Just make sure you have a recent python version. I used version 3.10 for the development.


## Usage

### Solving all problems
Use the following command to solve all the problems. I recommend to redirect the result in a file, ovtherwise it might overflow the terminal.
```
python mznb.py test -r problems
```

In the end, you should obtain something like the following.
```
mznb.py test -r problems
----------------------------------------
   OK   : problems/18_hole_golf.mzn
   OK   : problems/1d_rubiks_cube.mzn
   OK   : problems/30_bottles.mzn
   .
   .
   .
   OK   : problems/zoo_buses_and_kids.mzn
----------------------------------------
Start time: 2025-07-17 09:38:44.871379
End time  : 2025-07-17 09:44:43.364475
Duration  : 0:05:58.493096
----------------------------------------
INST-ERR:    0   0.0%
INST-TIM:    0   0.0%
COMP-ERR:    0   0.0%
COMP-TIM:    0   0.0%
SOLV-ERR:    0   0.0%
SOLV-TIM:    0   0.0%
CHEC-ERR:    0   0.0%
CHEC-TIM:    0   0.0%
   OK   :  766  100.0%
 TOTAL  :  766
```

### Solving a specific problems
To analyze what went wrong with a specific problem, use the following command.
```
python mznb.py test problems/model.mzn problems/data.dzn
```

The temp directory should now contain everything you need for the analysis.


## Reference

### Instance collection
As a general idea, an instance is either a single mzn file or a mzn-dzn pair. We proceed as following to collect the instances. Iterate over all input directories, recursively if `-r` option is used. Within a given directory, combine minizinc and datazinc files to form all mzn-dzn pairs. If the directory contains no datazinc file, every minizinc file is an instance.

All minizinc and datzinc files directly given as inputs are considered in a same virtual directory. Hence, they give one instance per mzn-dzn pair or one instance per mzn file if there is no dzn.

```
problems/
├── nqueens
│   ├── nqueens_v1.mzn
│   └── nqueens_v2.mzn
├── sudoku
│   ├── day1.dzn
│   ├── day2.dzn
│   ├── day3.dzn
│   ├── sudoku_v1.mzn
│   └── sudoku_v2.mzn
├── takuzu.dzn
└── takuzu.mzn
```

The directory structure above gives the following instances.
```
nqueens/nqueens_v1.mzn
nqueens/nqueens_v2.mzn
sudoku/sudoku_v1.mzn sudoku/day1.dzn
sudoku/sudoku_v1.mzn sudoku/day2.dzn
sudoku/sudoku_v1.mzn sudoku/day3.dzn
sudoku/sudoku_v2.mzn sudoku/day1.dzn
sudoku/sudoku_v2.mzn sudoku/day2.dzn
sudoku/sudoku_v2.mzn sudoku/day3.dzn
takuzu.mzn problems/takuzu.dzn
```


### Statuses
The statuses are defined as follows.
 - `INST-ERR` - the problem does not form a valid instance
 - `INST-TIM` - the instance verification timed out
 - `COMP-ERR` - an error occured during the compilation
 - `COMP-TIM` - the compilation timed out
 - `SOLV-ERR` - an error occured during the resolution
 - `SOLV-TIM` - the resolution timed out
 - `CHEC-ERR` - the solution is not correct
 - `CHEC-TIM` - the solution check timed out


## Documentation
The python script `mznb.py` is self documented using `argparse`. Use the following command to show the documentation.
```
python mznb.py --help
```

If you have any doubt about the way the script works, feel free to read the code. All functions and methods are documented.


## Versioning
Git tags are used to specify the dataset version. It is as easy as `v1`, `v2`, etc. Do not forget to add a new tag whenever you add, remove or update a problem.
```
git tag v3
```


## Useful links
 - [Hakank's MiniZinc page](https://www.hakank.org/minizinc/)
 - [MiniZinc documentation](https://docs.minizinc.dev/en/stable/index.html)
 - [MiniZinc playground](https://play.minizinc.dev/)
 - [Argparse documentation](https://docs.python.org/3/library/argparse.html)
