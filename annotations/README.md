# Annotations

This subfolder contains the scripts to

* evaluate the raw annotations from the trial and the main phase;
* analyze the final annotations; 
* visualize the final annotations.


## Evaluation

Get annotation evaluation statistics between the two annotators for each language pair:

For the trial annotations:

```
python scripts/evaluate_annotations.py annotations_trial/annotations1 annotations_trial/annotations2
```

For the main phase:
```
python scripts/evaluate_annotations.py annotations_raw/annotations1 annotations_raw/annotations2
```

## Annotation analysis

For statistics such as token count, minima, maxima and label distributions for SwissAdmin-RSD:

```
python scripts/get_stats.py gold_labels
```

Add the --ists flag for statistics on the iSTS-RSD dataset.


## Annotation visualization

For a color coded visualization of the SwissAdmin samples run:

```
python scripts/vis_diffs_in_sample.py gold_labels/gold_admin_[de/fr/it].jsonl --sample_id admin_[de/fr/it]_[0-224]
```

