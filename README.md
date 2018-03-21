WSDM Cup 2017 Vandalism Detection Task: Classification and Evaluation
=====================================================================

The [WSDM Cup 2017](https://www.wsdm-cup-2017.org/) was a data mining challenge held in conjunction with the 10th International Conference on Web Search and Data Mining (WSDM). The goal of the [vandalism detection task](https://www.wsdm-cup-2017.org/vandalism-detection.html) was to compute a vandalism score for each Wikidata revision denoting the likelihood of this revision being vandalism or similarly damaging. This is the classification and evaluation component for the baselines WDVD, ORES, and FILTER. The feature extraction can be done with the corresponding [feature extraction component](https://github.com/heindorf/wsdmcup17-wdvd-feature-extraction).

Paper
-----

This source code forms the basis for the overview paper of the [vandalism detection task at WSDM Cup 2017](https://arxiv.org/abs/1712.05956). When using the code, please make sure to refer to it as follows:

```TeX
@inproceedings{heindorf2017overview,
  author    = {Stefan Heindorf and
               Martin Potthast and
               Gregor Engels and
               Benno Stein},
  title     = {Overview of the Wikidata Vandalism Detection Task at {WSDM} Cup 2017},
  booktitle = {{{WSDM Cup 2017 Notebook Papers}},
  url       = {https://arxiv.org/abs/1712.05956},
  year      = {2017}
}
```

The code is based on the [Wikidata Vandalism Detector 2016](https://doi.acm.org/10.1145/2983323.2983740):

```TeX
@inproceedings{heindorf2016vandalism,
  author    = {Stefan Heindorf and
               Martin Potthast and
               Benno Stein and
               Gregor Engels},
  title     = {Vandalism Detection in Wikidata},
  booktitle = {{CIKM}},
  pages     = {327--336},
  publisher = {{ACM}},
  url       = {https://doi.acm.org/10.1145/2983323.2983740}
  year      = {2016}
}
```

Classification and Evaluation Component
---------------------------------------

### Requirements

The code was tested with Python 3.5.2, 64 Bit under Windows 10.

### Installation

We recommend [Miniconda](http://conda.pydata.org/miniconda.html) for easy installation on many platforms.

1. Create new environment: `conda create --name wsdmcup17 python=3.5.2 --file requirements.txt`
2. Activate environment: `activate wsdmcup17`
3. Copy the [AUCCalculator](http://mark.goadrich.com/programs/AUC/) to the folder `lib`

### Execute Classification

Usage:

    python wsdmcup17_classification.py FEATURES TRUTH RESULTS

Given a FEATURES file and TRUTH files (in bz2 format), splits the dataset, performs the classification and stores all results with the RESULTS prefix.

Example:

    python wsdmcup17_classification.py
        'features.csv.bz2'
        'wdvc-2016/training/wdvc16_truth.csv.bz2;wdvc-2016/validation/wdvc16_2016_03_truth.csv.bz2;wdvc-2016/testing/wdvc16_2016_05_truth.csv.bz2'
        'classification/20160101_0000000/20160101_0000000'

### Configure Evaluation

Configure the paths to the score files in the config file `teams.json`. For example,

    {
        "Buffaloberry": "wsdmcup17_buffaloberry.csv.bz2",
        "Conkerberry": "wsdmcup17_conkerberry.csv.bz2",
        "Honeyberry": "wsdmcup17_honeyberry.csv.bz2",
        "Loganberry": "wsdmcup17_loganberry.csv.bz2",
        "Riberry": "wsdmcup17_riberry.csv.bz2",
        "WDVD": "wsdmcup17_wdvd.csv.bz2",
        "ORES": "wsdmcup17_ores.csv.bz2",
        "FILTER": "wsdmcup17_filter.csv.bz2"
    }

### Execute Evaluation

Usage:

    python wsdmcup17_evaluation.py FEATURES TEAMS TRUTH RESULTS

Given a FEATURES file, a TEAMS file with paths to scores, a TRUTH files, and a RESULTS prefix, evaluates the performance of teams and computes meta approach.

Example:

    python wsdmcup17_evaluation.py
        'features.csv.bz2'
        'teams.json'
        'wdvc-2016/testing/wdvc16_2016_05_truth.csv.bz2'
        'evaluation/20160101_0000000/20160101_0000000'

### Configuration

The constants in the file config.py control what parts of the code are executed, the caching behavior as well as the level of parallelism.

Naturally, there is a tradeoff between maximum parallelism and minimum memory consumption. When executing all parts of the code with 16 parallel processes, about 256 GB RAM are required.

### Linting

Run `flake8`.

### Data Download

- Feature file as computed with the [feature extraction component](https://github.com/heindorf/wsdmcup17-wdvd-feature-extraction):
  - [wsdmcup17_features.csv.bz2](https://groups.uni-paderborn.de/wdqa/wsdmcup17/wsdmcup17_features.csv.bz2)
- Truth files from the [Wikidata Vandalism Corpus 2016](http://www.wsdm-cup-2017.org/vandalism-detection.html):
  - [wdvc16_truth.csv.bz2](https://groups.uni-paderborn.de/wdqa/wsdmcup17/wdvc16_truth.csv.bz2)
  - [wdvc16_2016_03_truth.csv.bz2](https://groups.uni-paderborn.de/wdqa/wsdmcup17/wdvc16_2016_03_truth.csv.bz2)
  - [wdvc16_2016_05_truth.csv.bz2](https://groups.uni-paderborn.de/wdqa/wsdmcup17/wdvc16_2016_05_truth.csv.bz2)
- Score files from the [WSDM Cup 2017 Proceedings](https://www.wsdm-cup-2017.org/proceedings.html):
  - [wsdmcup17_buffaloberry.csv.bz2](https://groups.uni-paderborn.de/wdqa/wsdmcup17/wsdmcup17_buffaloberry.csv.bz2)
  - [wsdmcup17_conkerberry.csv.bz2](https://groups.uni-paderborn.de/wdqa/wsdmcup17/wsdmcup17_conkerberry.csv.bz2)
  - [wsdmcup17_honeyberry.csv.bz2](https://groups.uni-paderborn.de/wdqa/wsdmcup17/wsdmcup17_honeyberry.csv.bz2)
  - [wsdmcup17_loganberry.csv.bz2](https://groups.uni-paderborn.de/wdqa/wsdmcup17/wsdmcup17_loganberry.csv.bz2)
  - [wsdmcup17_riberry.csv.bz2](https://groups.uni-paderborn.de/wdqa/wsdmcup17/wsdmcup17_riberry.csv.bz2)
  - [wsdmcup17_wdvd.csv.bz2](https://groups.uni-paderborn.de/wdqa/wsdmcup17/wsdmcup17_wdvd.csv.bz2)
  - [wsdmcup17_ores.csv.bz2](https://groups.uni-paderborn.de/wdqa/wsdmcup17/wsdmcup17_ores.csv.bz2)
  - [wsdmcup17_filter.csv.bz2](https://groups.uni-paderborn.de/wdqa/wsdmcup17/wsdmcup17_filter.csv.bz2)
  - [wsdmcup17_meta.csv.bz2](https://groups.uni-paderborn.de/wdqa/wsdmcup17/wsdmcup17_meta.csv.bz2)

Contact
-------

For questions and feedback please contact:

Stefan Heindorf, Paderborn University  
Martin Potthast, Leipzig University  
Gregor Engels, Paderborn University  
Benno Stein, Bauhaus-Universität Weimar

License
-------

The code by Stefan Heindorf, Martin Potthast, Gregor Engels, Benno Stein is licensed under a MIT license.
