import jsonlines
import os

def main():

    edit_sample_dict = {'de': 0, 'it': 0, 'fr': 0}
    edit_label_dict = {'de': 0, 'it': 0, 'fr': 0}
    total_label_dict = {'de': 0, 'it': 0, 'fr': 0}

    for edited, original in zip(os.listdir('edited'), os.listdir('original')):
        if edited.endswith('.jsonl') and original.endswith('.jsonl'):
            with jsonlines.open(f'edited/{edited}', 'r') as edited_reader, jsonlines.open(f'original/{original}', 'r') as original_reader:
                if "_de_" in edited and "_de_" in original:
                    lang = 'de'
                elif "_it_" in edited and "_it_" in original:
                    lang = 'it'
                elif "_fr_" in edited and "_fr_" in original:
                    lang = 'fr'
                else:
                    raise ValueError(f"Edited and original files do not have the same language: {edited} and {original}")
                
                for edited_item, original_item in zip(edited_reader, original_reader):
                    total_label_dict[lang] += len(edited_item['labels_a']) + len(edited_item['labels_b'])
                    if edited_item['labels_a'] != original_item['labels_a'] or edited_item['labels_b'] != original_item['labels_b']:
                        edit_sample_dict[lang] += 1
                        assert len(edited_item['labels_a']) == len(original_item['labels_a'])
                        assert len(edited_item['labels_b']) == len(original_item['labels_b'])

                        for edited_label, original_label in zip(edited_item['labels_a'], original_item['labels_a']):
                            if edited_label != original_label:
                                edit_label_dict[lang] += 1
                                

                        for edited_label, original_label in zip(edited_item['labels_b'], original_item['labels_b']):
                            if edited_label != original_label:
                                edit_label_dict[lang] += 1
                                
                    else:
                        continue

    print(f'Total number of edited samples: {edit_sample_dict}')
    print(f'Total number of edited labels: {edit_label_dict}')
    print(f'Total number of labels: {total_label_dict}')
    for lang in ['de', 'it', 'fr']:
        print(f'Percentage of edited labels in {lang}: {edit_label_dict[lang]/total_label_dict[lang]*100}')
        print(f'Percentage of edited samples in {lang}: {edit_sample_dict[lang]/50*100}')

if __name__ == "__main__":
    main()