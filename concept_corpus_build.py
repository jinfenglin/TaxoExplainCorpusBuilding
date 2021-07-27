import argparse
import os
import re
import ssl

from nltk import sent_tokenize

ssl._create_default_https_context = ssl._create_unverified_context
from queue import Queue
from time import sleep, time

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment
from cleantext import clean
from search_engines import Bing, Google
import jsonlines
from tqdm import tqdm
from multiprocessing.pool import ThreadPool

blacklist = [
    '[document]',
    'noscript',
    'header',
    'html',
    'meta',
    'head',
    'input',
    'script',
    'a',
    'style'
]


def clean_paragraph(doc):
    # add space around "-"
    doc = re.sub("-", " - ", doc)
    # merge multiple space
    doc = re.sub("\s+", " ", doc)
    # remove brackets with empty or non-alphbabatic  content
    doc = re.sub("[\[<\()][^a-zA-Z]+[\]\)>]", "", doc)
    # remove <EMAIL> and <URL>
    doc = re.sub("(<EMAIL>|<URL>)", "", doc)
    # split it into sentences
    sents = sent_tokenize(doc)
    # for each sentence strip - and space
    res = []
    for s in sents:
        for i, c in enumerate(s):
            if c.isalpha() and c.isupper():
                cs = s[i:]
                if len(cs.split()) > 3:
                    res.append(cs)
                break
    return [x.rstrip("\n\t\r -") for x in res]


def get_concepts(concept_file):
    cpts = set()
    if concept_file.endswith(".jsonl"):
        with jsonlines.open(concept_file) as fin:
            for obj in fin:
                cpts.update(obj['reduced_concepts'])
    elif concept_file.endswith(".csv"):
        cpt_df = pd.read_csv(concept_file)
        for idx, row in cpt_df.iterrows():
            art_cpts = eval(row["phrase"])
            cpts.update(art_cpts)

    elif concept_file.endswith(".txt"):
        with open(concept_file) as fin:
            for line in fin:
                cpts.add(line.strip("\n\t\r "))
    print(f"loaded {len(cpts)} concepts")
    return cpts


def is_valid_sentence(sent, query):
    def has_alphe(tk):
        for c in tk:
            if c.isalpha():
                return True
        return False

    sent, query = sent.lower(), query.lower()
    qtokens = [x for x in query.split() if has_alphe(x)]
    stokens = set([x for x in sent.split()])
    if len(stokens) < 5:
        return False
    cnt = 0
    for qtk in qtokens:
        if qtk in stokens:
            cnt += 1
    return cnt >= int(len(qtokens) * 0.5)


def scrap_worker(url, query, clean_lines):
    try:
        line_cnt_with_query = 0
        if url.endswith(".pdf") or url.endswith(".doc"):
            return
        header = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64)'}
        ca_path = 'G:\\Projects\\ASE_link_explain\\venv\\lib\\site-packages\\certifi\\cacert.pem'
        html = requests.get(url, headers=header, timeout=6, verify=ca_path).text
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.find_all(text=True)
        output = ''

        for t in text:
            if t.parent.name not in blacklist and not isinstance(t, Comment):
                output += '{} '.format(t)

        output = [y for y in [x.strip("\n\t\r ") for x in output.split("\n")] if len(y) > 0]

        for line in output:
            cline = clean(line).strip("\n\t\r ")
            sents = clean_paragraph(cline)
            for sent in sents:
                if line_cnt_with_query >= 80 or clean_lines.qsize() >= 500:
                    break
                if query.lower() in sent.lower():
                    line_cnt_with_query += 1
                else:
                    continue
                clean_lines.put(sent)
    except Exception as e:
        print(e)


def scrap_concept(qcpt, domain, page_num, visited_link):
    engine = Bing()
    clean_lines = Queue()
    if domain is None:
        query = f'what is "{qcpt}"'
    else:
        query = f'"{qcpt}" in {domain}'
    qres = engine.search(query, pages=page_num)
    links = [x for x in qres.links() if x not in visited_link][:45]
    visited_link.update(links)
    with ThreadPool(30) as p:
        p.starmap(scrap_worker, [(x, qcpt, clean_lines) for x in links])
    res = set()
    while not clean_lines.empty():
        res.add(clean_lines.get())
    return res


def concept_corpus_builder(concepts, out_file, page_num, domain, interval):
    visited_cpt, visited_link = set(), set()
    if os.path.isfile(out_file):
        with jsonlines.open(out_file) as fin:
            for obj in fin:
                visited_cpt.add(obj['query'])

    i, total = 0, len(concepts)
    with jsonlines.open(out_file, "a") as fout:
        print("start")
        for cpt in tqdm(concepts):
            start = time()
            i += 1
            print(f"{i}/{total}: {cpt}")
            if cpt in visited_cpt:
                continue
            visited_cpt.add(cpt)
            clean_lines = scrap_concept(qcpt=cpt, domain=domain, page_num=page_num,
                                        visited_link=visited_link)
            fout.write({
                "query": cpt,
                "sent_num": len(clean_lines),
                "sentences": list(clean_lines),
            })
            end = time()
            if end - start < 10:
                sleep(interval)


if __name__ == "__main__":
    "Collect related sentence from websites by providing it with concepts"
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept_file", help="path to the file store the concepts")
    parser.add_argument("--out_file", help="output the collected corpus in the format of jsonl")
    parser.add_argument("--page_num", default=5, type=int, help="the number of pages in search engine")
    parser.add_argument("--domain", default=None, help="the domain of the concepts if applicable")
    parser.add_argument("--query_interval", default=0.5, type=float)
    args = parser.parse_args()

    cpt_set = get_concepts(args.concept_file)
    concept_corpus_builder(concepts=cpt_set, out_file=args.out_file, page_num=args.page_num, domain=args.domain,
                           interval=args.query_interval)
