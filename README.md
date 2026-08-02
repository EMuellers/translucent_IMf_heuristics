# Heuristically Altering translucent Directly Follows Graphs
Here we provide the code for our thesis "". This work extends the translucent Inductive Miner-infrequent by Beyel et al.:

> **Harry H. Beyel, Wil M. P. van der Aalst**. (2025). *Using translucent activity relationships frequencies to enhance process discovery*.  Process Science 2, 15. [DOI](https://doi.org/10.1007/s44311-025-00010-y) | [Article Link](https://link.springer.com/article/10.1007/s44311-025-00010-y)

We only extended the IMtff and IMfts variants.
A variant can be chosen by setting the `translucent_variant` parameter in the parameters dict to either "IMtf" or "IMts"
To use the different heuristics, they should be provided as parameters to the algorithm. By default, all heuristics are considered to be off.
Here is an overview of the heuristics and which parameter needs to be set to use them:

## Delta Arc Heuristics
        
"remove_arcs_heuristics":   Remove arcs exclusive to tDFG before applying fall throughs, set to "False" (default) to disable. Possible values:     
                                # "dependency_score",  uses the dependency score metric to rank arcs
                                # "exclusive_choice_frequency" to use the choice frequency for ranking
                                # 'confidence' to use confidence of translucent df relation for ranking
                                # 'support' to use support of translucent df relation for ranking

"add_arcs_heuristics":      Add arcs from the tDFG to the DFG before applying fall throughs, set to "False" (default) to disable. Possible values:
                                # "support", # "dependency_score"
                                # "parallel_relationship_frequency" to use the choice frequency for ranking
                                # 'confidence' to use confidence of translucent df relation for ranking
                                # 'support' to use support of translucent df relation for ranking

## Single Activity Heuristics

"strict_end_activities": Either False or True, # Only consider translucent end activities which actually appear at the end of a trace at least once

"parallel_end_activities_heuristic": Either False or True, # If two activities are in translucent parallel relation and one is an end activity, the other is also considered an end activity in the (frequent) tDFG

"translucent_self_loops": Either False or True # Keep translucent self loops when projecting onto single activities, by projecting these on traces where activities follow themselves directly.

# Evaluation
All files pertaining to the evaluation can be found in the eval branch. Note that the version of the algorithm in the eval branch does not only return the pm4py Petri net but also the number of fallthroughs.

## Translucent precision
Our extension of the translucent precision measure to unfitting traces can be found in the "translucent_precision" directory.

## Translucent fitness
We ported the translucent fitness measure from networkx to rustworkx for better performance.
