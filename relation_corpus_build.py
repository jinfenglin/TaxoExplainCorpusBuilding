import argparse
import os
import re
import ssl

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
from nltk.tokenize import sent_tokenize

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


def get_links(link_file):
    links = []
    with open(link_file) as fin:
        for line in fin:
            cpts = line.strip("\n\t\r ").split(',')
            links.append((cpts[0], cpts[1]))
    return links


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


def scrap_worker(url, c1, c2, clean_lines):
    try:
        line_cnt_with_query = 0
        if url.endswith(".pdf") or url.endswith(".doc"):
            return
        header = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64)'}
        ca_path = 'G:\\Projects\\ASE_link_explain\\venv\\lib\\site-packages\\certifi\\cacert.pem'
        html = requests.get(url, headers=header, timeout=3, verify=ca_path).text
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
                if line_cnt_with_query >= 80 or clean_lines.qsize() >= 500 or len(sent.split()) > 75:
                    break
                if c1.lower() in sent.lower() and c2.lower() in sent.lower():
                    line_cnt_with_query += 1
                else:
                    continue
                clean_lines.put(sent)
    except Exception as e:
        print(e)


def scrap_concept(qcpt, page_num, visited_link):
    # engine = Google()
    engine = Bing()
    clean_lines = Queue()
    c1, c2 = qcpt[0].strip("\n\t\r "), qcpt[1].strip("\n\t\r ")
    # query = f'allintext:"{c1}" and "{c2}"'
    query = f'inbody:"{c1}" inbody:"{c2}"'
    qres = engine.search(query, pages=page_num)
    links = [x for x in qres.links() if x not in visited_link][:45]
    # visited_link.update(links)
    with ThreadPool(30) as p:
        p.starmap(scrap_worker, [(x, c1, c2, clean_lines) for x in links])
    res = set()
    while not clean_lines.empty():
        res.add(clean_lines.get())
    return res


def concept_corpus_builder(concepts, out_file, page_num, interval):
    visited_cpt, visited_link = set(), set()
    if os.path.isfile(out_file):
        with jsonlines.open(out_file) as fin:
            for obj in fin:
                query = obj['query']
                visited_cpt.add((query[0], query[1]))

    i, total = 0, len(concepts)
    with jsonlines.open(out_file, "a", flush=True) as fout:
        print("start")
        for cpt in tqdm(concepts):
            start = time()
            i += 1
            print(f"{i}/{total}: {cpt}")
            if cpt in visited_cpt:
                continue
            visited_cpt.add(cpt)
            clean_lines = set(scrap_concept(qcpt=cpt, page_num=page_num,
                                            visited_link=visited_link))
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
    parser.add_argument("--relation_file", help="path to the file store the concepts")
    parser.add_argument("--out_file", help="output the collected corpus in the format of jsonl")
    parser.add_argument("--page_num", default=2, type=int, help="the number of pages in search engine")
    parser.add_argument("--query_interval", default=0.5, type=float)
    args = parser.parse_args()

    cpt_set = get_links(args.relation_file)
    concept_corpus_builder(concepts=cpt_set, out_file=args.out_file, page_num=args.page_num,
                           interval=args.query_interval)
