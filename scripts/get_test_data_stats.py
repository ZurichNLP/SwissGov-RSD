import jsonlines
import json

import matplotlib.pyplot as plt
import numpy as np

"""with open('list_to_drop_junk.txt', 'r') as f:
    list_to_drop = f.readlines()
    list_to_drop = [line.strip() for line in list_to_drop]"""




with open('data/evaluation/gold_labels/full/gold.jsonl', 'r') as f, open('data/evaluation/gold_labels/full/gold_admin_de.jsonl', 'r') as ade, \
    open('data/evaluation/gold_labels/full/gold_admin_fr.jsonl', 'r') as afr, open('data/evaluation/gold_labels/full/gold_admin_it.jsonl', 'r') as ait:
    rsd = f.readlines()
    admin_de = ade.readlines()
    admin_fr = afr.readlines()
    admin_it = ait.readlines()

    def get_stats(data, lang=None):
        total_len = []

        for line in data:
            sample = json.loads(line)
            """if sample['id'] in list_to_drop:
                continue"""
            if 'ists' in sample['id']:
                if lang in sample['id']:
                    total_len.append((len(sample['text_a'])+len(sample['text_b']))/2)
                else:
                    continue
            else:
                #if len(sample['text_b']) < 400:
                    #print(f"{sample['id']}: {len(sample['text_a'])}: {sample['text_a']}")
                total_len.append((len(sample['text_a'])+len(sample['text_b']))/2)

        if 'ists' in sample['id']:
            assert len(total_len) == 100
        else:
            assert len(total_len) == 224

        min_length = min(total_len)
        max_length = max(total_len)
        avg_length = sum(total_len) / len(total_len)

        return min_length, max_length, avg_length

    rsd_de_min, rsd_de_max, rsd_de_avg = get_stats(rsd, 'de')
    rsd_fr_min, rsd_fr_max, rsd_fr_avg = get_stats(rsd, 'fr')
    rsd_it_min, rsd_it_max, rsd_it_avg = get_stats(rsd, 'it')
    admin_de_min, admin_de_max, admin_de_avg = get_stats(admin_de)
    admin_fr_min, admin_fr_max, admin_fr_avg = get_stats(admin_fr)
    admin_it_min, admin_it_max, admin_it_avg = get_stats(admin_it)

    print("Dataset    Min Length  Max Length  Avg Length")
    print("-" * 45)
    print(f'rsd_de:    {rsd_de_min:<10} {rsd_de_max:<10} {rsd_de_avg:.1f}')
    print(f'rsd_fr:    {rsd_fr_min:<10} {rsd_fr_max:<10} {rsd_fr_avg:.1f}')
    print(f'rsd_it:    {rsd_it_min:<10} {rsd_it_max:<10} {rsd_it_avg:.1f}')
    print(f'admin_de:  {admin_de_min:<10} {admin_de_max:<10} {admin_de_avg:.1f}')
    print(f'admin_fr:  {admin_fr_min:<10} {admin_fr_max:<10} {admin_fr_avg:.1f}')
    print(f'admin_it:  {admin_it_min:<10} {admin_it_max:<10} {admin_it_avg:.1f}')

    # Prepare data for box plots
    rsd_data = {
        'de': [rsd_de_min, rsd_de_avg, rsd_de_max],
        'fr': [rsd_fr_min, rsd_fr_avg, rsd_fr_max], 
        'it': [rsd_it_min, rsd_it_avg, rsd_it_max]
    }

    admin_data = {
        'de': [admin_de_min, admin_de_avg, admin_de_max],
        'fr': [admin_fr_min, admin_fr_avg, admin_fr_max],
        'it': [admin_it_min, admin_it_avg, admin_it_max]
    }
    # Set positions for box plots
    languages = ['de', 'fr', 'it']
    positions = np.arange(len(languages))
    width = 0.35

    # Create box plot
    fig, ax = plt.subplots(figsize=(6, 10))
    plt.rcParams.update({'font.size': plt.rcParams['font.size'] + 4})
    
    bp1 = ax.boxplot([rsd_data[lang] for lang in languages], 
                     positions=positions-width/2, 
                     widths=width,
                     patch_artist=True)
    bp2 = ax.boxplot([admin_data[lang] for lang in languages],
                     positions=positions+width/2,
                     widths=width, 
                     patch_artist=True)

    # Customize colors
    for box in bp1['boxes']:
        box.set(facecolor='blue', alpha=0.6)
    for box in bp2['boxes']:
        box.set(facecolor='orange', alpha=0.6)

    # Customize plot
    ax.set_xlabel('Languages', fontsize=plt.rcParams['font.size'])
    ax.set_ylabel('Number of Tokens', fontsize=plt.rcParams['font.size'])
    ax.set_xticks(positions)
    ax.set_xticklabels(languages, fontsize=plt.rcParams['font.size'])
    ax.tick_params(axis='y', labelsize=plt.rcParams['font.size'])
    ax.legend([bp1["boxes"][0], bp2["boxes"][0], bp1["medians"][0]], ['RSD', 'admin', 'Avg.'],
             loc='upper left', fontsize=plt.rcParams['font.size'])


    plt.savefig('data/evaluation/gold_labels/lengths.png')

            