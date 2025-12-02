import os
import json

aligned_pages = json.load(open("aligned_pages.json"))

print(f"Number of aligned pages in aligned_pages.json: {len(aligned_pages)}")
print(f'type of aligned_pages: {type(aligned_pages)}')

# check if each EN page has 3 aligned pages
for page, other_pages in aligned_pages.items():
    if len(other_pages) != 3:
        print(f"Page {page} has {len(other_pages)} aligned pages")

# check if each EN page has aligned pages in de, it, fr
for page, other_pages in aligned_pages.items():
    de = False
    fr = False
    it = False

    for aligned_page in other_pages:
        if "_de_" in aligned_page:
            de = True
        if "_fr_" in aligned_page:
            fr = True
        if "_it_" in aligned_page:
            it = True

    if not de:
        print(f"Page {page} has no aligned pages in de")
    if not fr:
        print(f"Page {page} has no aligned pages in fr")
    if not it:
        print(f"Page {page} has no aligned pages in it")
    
if de and fr and it:
    print(f"All pages have aligned pages in de, fr, it")

# check if a page has been aligned twice
check = []
dup_check = False
for page, other_pages in aligned_pages.items():
    for aligned_page in other_pages:
        if aligned_page in check:
            print(f"Page {aligned_page} has been aligned twice")
            dup_check = True
        check.append(aligned_page)

if not dup_check:
    print("No page has been aligned twice")

# check if all pages in cut_parsed_pages folders are in aligned_pages.json
cpp_check = False
for lang in ["de", "fr", "it"]:
    for page in os.listdir(f"cut_parsed_pages/{lang}"):
        page = page.replace(".txt", ".html")
        if page not in check:
            print(f"Page {page} is not in aligned_pages.json")
            cpp_check = True

if not cpp_check:
    print("All pages in cut_parsed_pages folders are in aligned_pages.json")


# check if all pages in aligned_pages.json are in cut_parsed_pages folder
folder_check = False
for lang in ["de", "fr", "it"]:
    for page, other_pages in aligned_pages.items():
        for aligned_page in other_pages:
            if f"_{lang}_" in aligned_page:
                aligned_page = aligned_page.replace(".html", ".txt")
                if aligned_page not in os.listdir(f"cut_parsed_pages/{lang}"):
                    print(f"Page {aligned_page} is not in cut_parsed_pages/{lang}")
                    folder_check = True

if not folder_check:
    print("All pages in aligned_pages.json are in cut_parsed_pages folders")



