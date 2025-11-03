# Translucent Activity Relationships
In this repository, we provide the code for our translucent discovery algorithm as well as the code for our precision method.
We also point out how we evaluated our method. We assume that enabled activities are provided with the attribute/column name: "enabled_activities".
For further information, and the data we used, we refer to our evaluation folder. 
## Create Translucent Event Logs from Existing Event Logs   
To create a translucent event log from an existing log, the following steps are necessary.
    
    import pm4py
    from synthetical_log_generation.log_generator import create_translucent_event_log
    from pm4py.objects.conversion.log import converter as log_converter

    initial_log = pm4py.read_xes('example.xes')
    initial_log = log_converter.apply(initial_log, variant=log_converter.Variants.TO_EVENT_LOG)
    net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(initial_log, noise_threshold=0.4)
    translucent_log = create_translucent_event_log(initial_log, net, initial_marking, final_marking)
    dataframe = pm4py.convert_to_dataframe(translucent_log)
    dataframe.to_csv('example_translucent.csv', sep=",", index=False)

## Discover Process Models from Translucent Event Logs
To discover process models, based on translucent event logs, the following steps are needed.
    
    import pm4py
    import pandas as pd
    from translucent_discovery.translucent_inductive_miner.translucent_base import discover_petri_net

    dataframe = pd.read_csv('running-example.csv', sep=";")
    dataframe = pm4py.format_dataframe(dataframe, case_id='CaseID', activity_key='Activity', timestamp_key='Timestamp')
    log = pm4py.convert_to_event_log(dataframe)
    model, i_m, f_m = discover_petri_net(log, {"translucent_variant": "IMts"})

Note that there are different parameters for `translucent_variant` possible. In particular, the following: 
`IM`, `IMto`, `IMtf`, and `IMts`.

Other newly added parameters concern noise filtering and which graph to use when applying fall-throughs.

    import pm4py
    import pandas as pd
    from translucent_discovery.translucent_inductive_miner.translucent_base import discover_petri_net

    dataframe = pd.read_csv('running-example.csv', sep=";")
    dataframe = pm4py.format_dataframe(dataframe, case_id='CaseID', activity_key='Activity', timestamp_key='Timestamp')
    log = pm4py.convert_to_event_log(dataframe
    noise_filter = 0.2
    model, i_m, f_m = discover_petri_net(log, {"translucent_variant": "IMts",  "tDFG_fall_through": False}, 0.2))

The value for the parameter `tDFG_fall_through` is either `True` or `False`.


## Translucent Precision 
To compute a translucent precision score, the following is needed. 
    
    from translucent_precision.main import translucent_precision_score
    import pm4py
    import pandas as pd
    from translucent_discovery.translucent_inductive_miner.translucent_base import discover_petri_net
    
    dataframe = pd.read_csv('running-example.csv', sep=";")
    dataframe = pm4py.format_dataframe(dataframe, case_id='CaseID', activity_key='Activity', timestamp_key='Timestamp')
    log = pm4py.convert_to_event_log(dataframe)
    model, i_m, f_m = discover_petri_net(log, {"translucent_variant": "IMts"})
    translucent_precision_score(log, model, i_m, f_m)

## Evaluation of our work

For our evaluation, the necessary files are located in the evaluation folder. Each scenario has its own folder. 
Next, we explain the structure for the artificial scenario and the real-life scenario.

### Artificial Scenario
The artificial scenario consists of the file `evaluate.py` file and the different event logs. The number states the number of variants.

### Real-life Scenario
Each folder consists of `evaluate.py`. This file creates in the corresponding sub-folders the associated ground truth and the different variants.
In the "04" folders, the data for the experiments in our paper are stored.
The original events logs are public available and are not part of this Github due to their size.

### Publications
This work has been published. If you find the methods and datasets useful, please consider citing the following paper(s): 

> **Harry H. Beyel, Wil M. P. van der Aalst**. (2024). *Improving Process Discovery Using Translucent Activity Relationships*. Business Process Management, pp.146-163. [DOI](https://doi.org/10.1007/978-3-031-70396-6_9) | [Article Link](https://link.springer.com/chapter/10.1007/978-3-031-70396-6_9)

BiBTeX: 
```bibtex
@inproceedings{TranslucentActivityRelationships,
  author       = {Harry H. Beyel and
                  Wil M. P. van der Aalst},
  editor       = {Andrea Marrella and
                  Manuel Resinas and
                  Mieke Jans and
                  Michael Rosemann},
  title        = {Improving Process Discovery Using Translucent Activity Relationships},
  booktitle    = {Business Process Management - 22nd International Conference, {BPM}
                  2024, Krakow, Poland, September 1-6, 2024, Proceedings},
  series       = {Lecture Notes in Computer Science},
  volume       = {14940},
  pages        = {146--163},
  publisher    = {Springer},
  year         = {2024},
  url          = {https://doi.org/10.1007/978-3-031-70396-6\_9},
  doi          = {10.1007/978-3-031-70396-6\_9},
}
```


> **Harry H. Beyel, Wil M. P. van der Aalst**. (2025). *Using translucent activity relationships frequencies to enhance process discovery*.  Process Science 2, 15. [DOI](https://doi.org/10.1007/s44311-025-00010-y) | [Article Link](https://link.springer.com/article/10.1007/s44311-025-00010-y)

BiBTeX: 
```bibtex
@inproceedings{TranslucentActivityRelationshipsFrequency,
  author       = {Harry H. Beyel and
                  Wil M. P. van der Aalst},
  journal      = {Process Science},
  volume       = {2},
  number       = {1},
  year         = {2025},
  doi          = {10.1007/s44311-025-00010-y},
}
```

