import jsonlines
import argparse as ap

def main():
    parser = ap.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Path to input file of containing the final annotations.")
    parser.add_argument("--sample_id", type=str, help="Sample id to investigate only one specific document pair.")
    args = parser.parse_args()

    with jsonlines.open(args.input_file) as reader:
        for item in reader:
            if args.sample_id and args.sample_id == item['id']:
                
                assert len(item['text_a'].split()) == len(item['labels_a']), f"Length of text_a: {len(item['text_a'].split())} and labels_a: {len(item['labels_a'])} do not match for {item['id']}"
                assert len(item['text_b'].split()) == len(item['labels_b']), f"Length of text_b: {len(item['text_b'].split())} and labels_b: {len(item['labels_b'])} do not match for {item['id']}"
                print("English side:")

                # Define colors for different labels
                colors = {
                    -1.0: "\033[91m",  # Red
                    0.0: "",           # No color
                    0.2: "\033[92m",   # Green
                    0.4: "\033[95m",   # Magenta
                    0.6: "\033[93m",   # Yellow
                    0.8: "\033[96m",   # Cyan
                    1.0: "\033[94m"    # Blue
                }

                # Print legend
                print("\nLegend:")
                for label, color in colors.items():
                    print(f"{color}■\033[0m : {label}")
                print()

                # Print English text with colors
                print("English side:")
                for i in range(len(item['text_a'].split())):
                    label = item['labels_a'][i]
                    word = item['text_a'].split()[i]
                    print(f"{colors[label]}{word}\033[0m", end=" ")

                print()

                print("Non-English side:")
                for i in range(len(item['text_b'].split())):
                    label = item['labels_b'][i]
                    word = item['text_b'].split()[i]
                    print(f"{colors[label]}{word}\033[0m", end=" ")

                print()
                
                

if __name__ == "__main__":
    main()