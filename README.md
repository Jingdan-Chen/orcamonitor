## orcamonitor

1. Orca6 outputs the imaginary mode at every step for TS optimization tasks (likely the internal coordinates with larger contributions corresponding to the current imaginary frequency vibration)
2. When there is a `recalc_Hess` during an optimization task, OrcaMonitor cannot extract the corresponding information.
    PS: I noticed that even with Orca updated to version 6, there are still some issues with geometry optimization worth criticizing. The geometric optimization in Gaussian is more prone to extra imaginary frequencies, but setting `opt` to `convergence tight` often results in cases where `calc_Hess` already shows the correct number of imaginary frequencies, and yet the geometry optimization cannot converge despite meeting several convergence criteria.

To address this problem, I rewrote my own version of orcamonitor using Python 3. It requires NumPy installed.

```bash
$ ./orcamonitor.py -h
usage: orcamonitor.py [-h] [-f FREQ_TYPE] [-i] [-x XYZ] [-o] filename

Process ORCA output files.

positional arguments:
  filename              The ORCA output file to process

options:
  -h, --help            show this help message and exit
  -f FREQ_TYPE, --freq_type FREQ_TYPE
                        Frequency type: opt or ts (default: opt)
  -i, --interactive     Enable interactive mode
  -x XYZ, --xyz XYZ     Extract the xyz frame
  -o, --ongoing         never raise error if encoutered
```

**PS:**

1. The default output of the five geometry optimization convergence criteria is given as the ratio of the current value to the "normal" convergence limit (i.e., <1 indicates that this criterion has met the normal convergence condition).
2. `MonConv` evaluates the current iteration's imaginary frequency (`im_freq`) and the five convergence criteria. It outputs "YES" only when the following two conditions are met:
   1. The `im_freq` (from either `recalc_Hess` or the frequency calculation at the end of optimization) is non-empty, and the number of imaginary frequencies meets the optimization type requirement (`opt` requires 0 imaginary frequencies, and `optTS` requires 1 imaginary frequency).
   2. Four of the five geometry optimization convergence criteria are satisfied, *or* the energy change has converged, and two gradient convergence limits have reached half the "normal" standard (see the code for specifics).

The reason for designing this peculiar behavior is that, during my calculations, I often use scripts to monitor outputs in real-time. As soon as `recalc_Hess` shows the correct number of imaginary frequencies and the five optimization criteria are nearly met, I manually intervene in ORCA to improve efficiency.





