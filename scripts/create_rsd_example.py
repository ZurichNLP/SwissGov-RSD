MAX_COLOR = 40

gpt_default_predictions = {
  "sentence1": [["Top", 2], ["Diplomats", 5], ["Meet", 1], ["in", 0], ["Munich", 3]],
  "sentence2": [["De", 0], ["hauts", 0], ["diplomates", 5], ["discutent", 2], ["du", 0], ["programme", 0], ["nucléaire", 0]]
}

gpt_finetuned_predictions = {
  "sentence1": [["Top", 5], ["Diplomats", 5], ["Meet", 3], ["in", 0], ["Munich", 0]],
  "sentence2": [["De", 5], ["hauts", 5], ["diplomates", 5], ["discutent", 3], ["du", 0], ["programme", 0], ["nucléaire", 0]]
}

modernbert_prediction = """DifferenceSample(tokens_a=('Top', 'Diplomats', 'Meet', 'in', 'Munich'), tokens_b=('De', 'hauts', 'diplomates', 'discutent', 'du', 'programme', 'nucléaire'), labels_a=(0.17933662235736847, 0.21594615777333578, 0.9875600934028625, 0.9776043891906738, 0.9676840305328369), labels_b=(0.31173399090766907, 0.28604479134082794, 0.31815143674612045, 0.9549896518389384, 0.9509885311126709, 0.9844472408294678, 0.9133208841085434)"""

def format_tokens(tokens, map_fn):
    custom_color = "berry"
    # Concatenate LaTeX colorbox commands for each token using the provided mapping function.
    return "".join(f"\\colorbox{{{custom_color}!{map_fn(value):02d}}}{{\\strut {token}}}" for token, value in tokens)

def normalize_scores(scores):
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [0] * len(scores)
    return [(s - min_score) / (max_score - min_score) for s in scores]

# Mapping functions for the two scales.
gpt_map = lambda v: int(round((5 - v) / 5 * MAX_COLOR))
modernbert_map = lambda v: int(round(v * MAX_COLOR))

# Get raw color values for GPT predictions
gpt_default_colors1 = [gpt_map(score) for _, score in gpt_default_predictions["sentence1"]]
gpt_default_colors2 = [gpt_map(score) for _, score in gpt_default_predictions["sentence2"]]
gpt_default_all_colors = gpt_default_colors1 + gpt_default_colors2
normalized_gpt_default = normalize_scores(gpt_default_all_colors)

# Create normalized predictions
gpt_default_norm = {
    "sentence1": list(zip([t[0] for t in gpt_default_predictions["sentence1"]], normalized_gpt_default[:len(gpt_default_colors1)])),
    "sentence2": list(zip([t[0] for t in gpt_default_predictions["sentence2"]], normalized_gpt_default[len(gpt_default_colors1):]))
}

# Get raw color values for GPT finetuned predictions
gpt_finetuned_colors1 = [gpt_map(score) for _, score in gpt_finetuned_predictions["sentence1"]]
gpt_finetuned_colors2 = [gpt_map(score) for _, score in gpt_finetuned_predictions["sentence2"]]
gpt_finetuned_all_colors = gpt_finetuned_colors1 + gpt_finetuned_colors2
normalized_gpt_finetuned = normalize_scores(gpt_finetuned_all_colors)

# Create normalized predictions
gpt_finetuned_norm = {
    "sentence1": list(zip([t[0] for t in gpt_finetuned_predictions["sentence1"]], normalized_gpt_finetuned[:len(gpt_finetuned_colors1)])),
    "sentence2": list(zip([t[0] for t in gpt_finetuned_predictions["sentence2"]], normalized_gpt_finetuned[len(gpt_finetuned_colors1):]))
}

# Parse the modernbert_prediction string.
mp = modernbert_prediction
# Extract tokens for sentence1.
start = mp.find("tokens_a=(")
end = mp.find("), tokens_b=(")
tokens_a_str = mp[start + len("tokens_a=("): end]
tokens_a = [s.strip().strip("'") for s in tokens_a_str.split(",")]

# Extract tokens for sentence2.
start_b = end + len("), tokens_b=(")
end_b = mp.find("), labels_a=(")
tokens_b_str = mp[start_b: end_b]
tokens_b = [s.strip().strip("'") for s in tokens_b_str.split(",")]

# Extract labels for sentence1.
start_la = end_b + len("), labels_a=(")
end_la = mp.find("), labels_b=(")
labels_a_str = mp[start_la: end_la]
labels_a = [float(s.strip()) for s in labels_a_str.split(",")]

# Extract labels for sentence2.
start_lb = end_la + len("), labels_b=(")
end_lb = mp.rfind(")")
labels_b_str = mp[start_lb: end_lb]
labels_b = [float(s.strip()) for s in labels_b_str.split(",")]

# Get raw color values for ModernBERT predictions
modernbert_colors1 = [modernbert_map(score) for score in labels_a]
modernbert_colors2 = [modernbert_map(score) for score in labels_b]
modernbert_all_colors = modernbert_colors1 + modernbert_colors2
normalized_modernbert = normalize_scores(modernbert_all_colors)

modernbert_preds_sentence1 = list(zip(tokens_a, normalized_modernbert[:len(labels_a)]))
modernbert_preds_sentence2 = list(zip(tokens_b, normalized_modernbert[len(labels_a):]))

# Mapping function for the normalized scale
score_to_color = lambda v: int(round(v * MAX_COLOR))

# Format the predictions
gpt_default_line1 = format_tokens(gpt_default_norm["sentence1"], score_to_color)
gpt_default_line2 = format_tokens(gpt_default_norm["sentence2"], score_to_color)

gpt_finetuned_line1 = format_tokens(gpt_finetuned_norm["sentence1"], score_to_color)
gpt_finetuned_line2 = format_tokens(gpt_finetuned_norm["sentence2"], score_to_color)

modernbert_line1 = format_tokens(modernbert_preds_sentence1, score_to_color)
modernbert_line2 = format_tokens(modernbert_preds_sentence2, score_to_color)

# Create the filled LaTeX template.
filled_template = r"""
\definecolor{berry}{HTML}{B12840}
\scalebox{0.88}{%
\begin{minipage}{\textwidth}
{\small \textbf{\textsf{GPT-4o-mini, few-shot prompting}}}

\vspace{0.1cm}
{\small \textit{\textsf{""" + gpt_default_line1 + r"""}}}

\vspace{0.1cm}
{\small \textit{\textsf{""" + gpt_default_line2 + r"""}}}

\vspace{0.4cm}
{\small \textbf{\textsf{GPT-4o-mini, fine-tuned}}}

\vspace{0.1cm}
{\small \textit{\textsf{""" + gpt_finetuned_line1 + r"""}}}

\vspace{0.1cm}
{\small \textit{\textsf{""" + gpt_finetuned_line2 + r"""}}}

\vspace{0.4cm}
{\small \textbf{\textsf{ModernBERT, fine-tuned}}}

\vspace{0.1cm}
{\small \textit{\textsf{""" + modernbert_line1 + r"""}}}

\vspace{0.1cm}
{\small \textit{\textsf{""" + modernbert_line2 + r"""}}}
\end{minipage}
}"""

print(filled_template)
