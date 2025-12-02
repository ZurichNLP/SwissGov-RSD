from pathlib import Path

import numpy as np
from nlpstats.correlations import correlate, bootstrap

from evaluation.utils import load_predictions, load_gold_data


def main(predictions_path: Path, subset: str = None):
    predictions = load_predictions(predictions_path)

    gold_path = Path(__file__).parent.parent / 'data' / 'evaluation' / 'gold_labels' / 'full' / 'gold.jsonl'
    gold_samples = load_gold_data(gold_path)

    assert len(predictions) == len(gold_samples)

    benchmark_names = [
        "ists",
        "ists_negatives",
        "ists_documents", 
        "ists_permutations",
        "ists_de",
        "ists_es",
        "ists_fr", 
        "ists_ja",
        "ists_ko",
        "ists_zh",
        "ists_it",  
    ]

    if subset:
        if subset not in benchmark_names:
            raise ValueError(f"Invalid subset name. Must be one of: {benchmark_names}")
        benchmarks_to_evaluate = [subset]
    else:
        benchmarks_to_evaluate = benchmark_names

    # Calculate correlation for each subset
    all_correlations = []
    all_intervals = []
    for benchmark in benchmarks_to_evaluate:
        i = benchmark_names.index(benchmark)
        start_idx = i * 100
        end_idx = (i + 1) * 100
        
        subset_predictions = predictions[start_idx:end_idx]
        subset_gold = gold_samples[start_idx:end_idx]
        assert len(subset_predictions) == len(subset_gold) == 100

        pred_labels = []
        gold_labels = []
        counter = 0
        counter_b = 0
        for prediction, gold_sample in zip(subset_predictions, subset_gold):
            #assert prediction.sentence1 == " ".join(gold_sample.tokens_a), f"{prediction.sentence1} != {' '.join(gold_sample.tokens_a)}"
            #assert prediction.sentence2 == " ".join(gold_sample.tokens_b), f"{prediction.sentence2} != {' '.join(gold_sample.tokens_b)}"
            pred_labels_a = prediction.get_difference_sample().labels_a
            if len(pred_labels_a) != len(gold_sample.labels_a):
                pred_labels_a = pred_labels_a + (0,) * (len(gold_sample.labels_a) - len(pred_labels_a))
                counter += 1
            gold_labels_a = gold_sample.labels_a
            assert len(pred_labels_a) == len(gold_labels_a), f"{len(pred_labels_a)} != {len(gold_labels_a)}"
            pred_labels.extend(pred_labels_a)
            gold_labels.extend(gold_labels_a)

            gold_labels_b = gold_sample.labels_b
            if not all(label == -1 for label in gold_labels_b):
                pred_labels_b = prediction.get_difference_sample().labels_b
                if len(pred_labels_b) != len(gold_labels_b):
                    pred_labels_b = pred_labels_b + (0,) * (len(gold_labels_b) - len(pred_labels_b))
                    counter_b += 1
                assert len(pred_labels_b) == len(gold_labels_b), f"{len(pred_labels_b)} != {len(gold_labels_b)}"
                pred_labels.extend(pred_labels_b)
                gold_labels.extend(gold_labels_b)

        # Filter out labels where gold is -1
        filtered_pred_labels = [pred for pred, gold in zip(pred_labels, gold_labels) if gold != -1]
        filtered_gold_labels = [gold for pred, gold in zip(pred_labels, gold_labels) if gold != -1]
        print(f"Number of samples that did not have the label length a: {counter}")
        print(f"Number of samples that did not have the label length b: {counter_b}")

        assert len(filtered_pred_labels) == len(filtered_gold_labels)

        correlation = correlate(
            np.expand_dims(np.array(filtered_pred_labels), 0),
            np.expand_dims(np.array(filtered_gold_labels), 0),
            level="global",
            coefficient="spearman",
        )
        bootstrap_result = bootstrap(
            np.expand_dims(np.array(filtered_pred_labels), 0),
            np.expand_dims(np.array(filtered_gold_labels), 0),
            level="global",
            coefficient="spearman",
            resampling_method="inputs",
            #n_resamples=1,
        )
        max_interval = max(bootstrap_result.upper - correlation, correlation - bootstrap_result.lower)
        print(f"{benchmark}: {correlation*100:.1f} ±{max_interval*100:.1f}")

        all_correlations.append(correlation)
        all_intervals.append(max_interval)

    if not subset:
        avg_cross_lingual_correlation = np.mean(all_correlations[4:])
        avg_cross_lingual_interval = np.mean(all_intervals[4:])
        print(f"\nAvg cross-lingual: {avg_cross_lingual_correlation*100:.1f} ±{avg_cross_lingual_interval*100:.1f}")

        print(predictions_path.stem + " & " + " & ".join(f"{c*100:.1f} \ci{{{ci*100:.1f}}}" for c, ci in zip(
            all_correlations[:4] + [avg_cross_lingual_correlation],
            all_intervals[:4] + [avg_cross_lingual_interval]
        )))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('predictions_path', type=Path)
    parser.add_argument('--subset', type=str, help='Evaluate only a specific subset')
    args = parser.parse_args()
    main(args.predictions_path, args.subset)