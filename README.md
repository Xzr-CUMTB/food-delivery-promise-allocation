# Food-delivery promise allocation experiments

Reconstruct fixed rider cohorts, allocate promise extensions and check the complete pickup-and-delivery route. All time quantities are in minutes. Python 3.13 was used for the revision.

Obtain the official Meituan files `all_waybill_info_meituan_0322.csv.zip` and `dispatch_rider_meituan.csv` from [Meituan–INFORMS TSL](https://github.com/meituan/Meituan-INFORMS-TSL-Research-Challenge). Place both files in one directory. Their use is governed by the provider's License.txt. This research was supported by data provided by Meituan.

```sh
python -m pip install -r requirements.txt
python run.py --source /path/to/official-files --output /path/to/run-directory
```

The program reconstructs active task states, builds the internal prediction screen, solves paired allocation scenarios, enumerates the small exact benchmarks, computes rider-cluster intervals, and draws the four revision figures. Generated tables are in `tables`. The first complete run includes Numba initialization. Timings depend on hardware, load and compiled-function cache. Times in the paper refer to the original revision run.

The reference case uses 3 minutes of initial rest, 15 km/h travel, 0.5 minutes of service per node, 3 order-minutes per order and a 6-minute individual cap. Expected reference counts are 835 certified-positive cohorts under uniform allocation, 565 under capped individual targeting, 512 under joint-route targeting and 470 without a cap. The outcome is computed on 13,772 fixed cohorts.

For each route, the solver evaluates a common deadline shift using a precedence-constrained subset dynamic program. It retains the earliest arrival for each visited set and endpoint. The exact resource comparison uses exhaustive permutation search for selected two- to four-order states. The share-positive rule uses the lower bound and retains threshold-straddling intervals. Additional settings, limits, tie rules and figure data are defined in the corresponding scripts.
