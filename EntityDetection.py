import os
import pathlib
import sys
from collections import defaultdict, OrderedDict

import stanza
import pandas as pd
import logging

from stanza.server import CoreNLPClient, StartServer
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger(__name__)

month_set = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "augest",
    "september",
    "october",
    "november",
    "december",
    "jan",
    "feb",
    "apr",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
}

num_set = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
}

adj_blk_list = {
    "other",
    "more",
    "few",
    "some",
    "many",
    "else",
    "new",
    "old",
    "good",
    "bad",
    "nice",
}


class Concept:
    def __init__(self, start, end, text):
        self.start, self.end = start, end  # start and end index of tokens
        self.text = text

    def __str__(self):
        return self.text


class Relation:
    def __init__(self, tokens, sent_index):
        self.tokens = tokens
        self.sent_index = sent_index
        self.rel = dict()  # key is subject, value is object


class DomainKG:
    def __init__(self, stop_words=None, start_server=StartServer.TRY_START):
        if stop_words is None:
            stop_words = set()
        self.client = CoreNLPClient(
            annotators=["tokenize", "ssplit", "lemma", "pos", "depparse"],
            timeout=30000,
            memory="8G",
            properties={"depparse.extradependencies": "MAXIMAL"},
            start_server=start_server,
            preload=True,
            be_quiet=True,
        )
        self.client.annotate("")
        self.stop_words = set(stop_words)
        self.concepts, self.relations = [], []

    def extend_adj(self, start, words, pos):
        if start > 0 and pos[start - 1].startswith("JJ"):  # first tk must be adj
            s = start = start - 1
            while start >= 0:
                cur_pos = pos[start]
                cur_word = words[start]
                if (
                        cur_pos.startswith("JJ") and cur_word not in self.stop_words
                ):  # save the last JJ
                    s = start
                else:
                    break
                start -= 1
            return s
        return start

    @staticmethod
    def extract_concepts(words, pos):
        """
        Extract the concepts based on the postags. It will extend the concepts with adjectives and conjunctions
        :param words:
        :param pos:
        :return:
        """

        def valid_token_for_phrase(cur_w, cur_p, next_w, next_p):
            valid_tags = {"IN", "NN", "HYPH", "CD"}
            valid_text = {"\\", "/"}
            if cur_w in valid_text:
                return True
            for t in valid_tags:
                if cur_p.startswith(t):
                    return True
            if cur_p.startswith("JJ") and next_w and next_p.startswith("NN"):
                return True
            return False

        def valid_leading_token(cur_w, cur_p):
            if cur_w in num_set or cur_w in month_set or cur_w in adj_blk_list:
                return False

            contain_alph = False
            for i, c in enumerate(cur_w):
                if c.isalpha():
                    contain_alph = True
                    break
                elif i == 0:  # leading token must start with character
                    return False

            if not contain_alph:
                return False
            return True

        concepts = []
        i, N = 0, len(words)
        while i < N:
            if pos[i].startswith("NN") and valid_leading_token(words[i], pos[i]):  # start with NN
                s, e, j = i, i + 1, i + 1
                while j < N:
                    j_pos = pos[j]
                    next_w = words[j + 1] if j + 1 < N else None
                    next_p = pos[j + 1] if j + 1 < N else None

                    if j_pos.startswith("NN"):
                        e = j + 1
                    if not valid_token_for_phrase(words[j], j_pos, next_w, next_p):
                        break
                    j += 1
                i = j

                ns = s  # FIXME try not extend the phrases to reduce the error rate
                # ns = self.extend_adj(s, words, pos)
                # keep only the noun part if the leading token is not valid
                # if not valid_leading_token(words[ns], pos[ns]):
                #     ns = s
                c = Concept(ns, e, " ".join([x for x in words[ns:e]]))
                concepts.append(c)
            i += 1
        return concepts

    @staticmethod
    def expand_verb(sent, idx):
        l, r = idx - 1, idx + 1
        ltk, rtk = [], []
        exp_tag = ["IN", "RP", "TO", "VB", "CC"]
        while l >= 0:
            if sent.token[l].pos[:2] in exp_tag:
                ltk.append(sent.token[l].word)
            else:
                break
            l -= 1
        while r < len(sent.token):
            if sent.token[r].pos[:2] in exp_tag:
                rtk.append(sent.token[r].word)
            else:
                break
            r += 1
        exp_verb = ltk[:-1] + [sent.token[idx].word] + rtk[:]
        return " ".join(exp_verb)

    @staticmethod
    def extract_relations(sent, doc_concepts):
        def to_rel(c1, verb, c2):
            c1_str = str(doc_concepts.get(c1, ""))
            c2_str = str(doc_concepts.get(c2, ""))
            if c1_str == c2_str or c1_str == "" or c2_str == "":
                return None
            lmtzer = WordNetLemmatizer()
            verb.replace("_", " ")
            verb = " ".join([lmtzer.lemmatize(x) for x in verb.split()])
            return (c1_str, verb, c2_str)

        def add_relation(rel_set, c1, verb, c2):
            relation = to_rel(c1, verb, c2)
            if relation:
                rel_set.add(relation)

        doc_rels = set()
        in_deps = defaultdict(dict)
        out_deps = defaultdict(dict)

        for r in sent.enhancedPlusPlusDependencies.edge:
            # abandon the case annotation of the dep
            t1_idx, rel, t2_idx = r.source - 1, r.dep.split(":"), r.target - 1
            t1_pos, t2_pos = sent.token[t1_idx].pos, sent.token[t2_idx].pos
            rcase = rel[1] if len(rel) > 1 else ""
            rel = rel[0]

            # if t1_pos.startswith("NN") and t1_idx in doc_concepts and t2_pos.startswith(
            #         "NN") and t2_idx in doc_concepts and rcase is not "":
            #     add_relation(doc_rels, t1_idx, rcase, t2_idx)
            out_rels = out_deps[t1_idx]
            tmp = out_rels.get(rel, [])
            tmp.append(t2_idx)
            out_deps[t1_idx][rel] = tmp

            in_rels = in_deps[t2_idx]
            tmp = in_rels.get(rel, [])
            tmp.append(t1_idx)
            in_deps[t2_idx][rel] = tmp

        for idx, w in enumerate(sent.token):
            if w.pos.startswith("VB"):
                # sub-obj rule
                expand_vb = DomainKG.expand_verb(sent, idx)
                subjs = out_deps[idx].get("nsubj", [])
                if len(subjs) == 0:
                    subjs = in_deps[idx].get("acl", [])
                if len(subjs) == 0:
                    xcomp_dep = in_deps[idx].get("xcomp", [])
                    if len(xcomp_dep) > 0:
                        comp_vb = out_deps[xcomp_dep[0]]
                        subjs = comp_vb.get("obj", []) + comp_vb.get("obl", [])
                for subj in subjs:
                    if subj not in doc_concepts:
                        continue
                    objs = out_deps[idx].get("obj", [])
                    if len(objs) == 0:
                        objs = out_deps[idx].get("obl", [])
                    for obj in objs:
                        if obj not in doc_concepts:
                            continue
                        add_relation(doc_rels, subj, expand_vb, obj)
                    break  # pick only one subj
            elif w.pos.startswith("NN"):
                # "[IN]-> case ->[NN] -> obl-> [VB]-> obj-> [NN]"
                case_dep = out_deps[idx].get("case", [])
                obl_dep = in_deps[idx].get("obl", [])
                if len(case_dep) > 0 and len(obl_dep) > 0:
                    vb = DomainKG.expand_verb(sent, case_dep[0])
                    objs = out_deps[obl_dep[0]].get("obj", [])
                    if len(objs) > 0:
                        obj = objs[0]
                        if sent.token[obj].pos.startswith("NN"):
                            add_relation(doc_rels, obj, vb, idx)
        return doc_rels

    def build(self, docs, disable_tqdm=False):
        concepts = []  # concepts in each document as a list of list
        rels = []  # concept in each document as a list of list
        tokens = []
        for d in tqdm(docs, disable=disable_tqdm):
            doc_concepts = dict()  # list of concepts in a document
            doc_rels = []  # list of relations in a document
            doc_tokens = []
            for i, sent in enumerate(self.client.annotate(d).sentence):
                try:
                    words, pos = [], []
                    for w in sent.token:
                        words.append(w.word)
                        pos.append(w.pos)

                    for c in self.extract_concepts(words, pos):
                        for i in range(c.start, c.end):
                            doc_concepts[i] = c
                    doc_rels.extend(self.extract_relations(sent, doc_concepts))
                    doc_tokens.extend(words)
                except Exception as e:
                    raise Exception(e)

            concepts.append([str(x) for x in set(doc_concepts.values())])
            rels.append(doc_rels)
            tokens.append(doc_tokens)

        return concepts, rels, tokens


def process_CM1():
    s_cm1, t_cm1 = pd.read_csv("./data/CM1/source_artifacts.csv"), pd.read_csv("./data/CM1/target_artifacts.csv")
    cm1 = s_cm1.append(t_cm1)
    arts = cm1["arts"].to_list()
    ids = cm1["id"].to_list()
    out_dir = "./output/CM1/"

    dkg = DomainKG()
    concepts, rels, tokens = dkg.build(arts)
    concept_df = pd.DataFrame()
    concept_df["phrase"] = concepts
    concept_df["ids"] = ids
    concept_df.to_csv(os.path.join(out_dir, "concepts.csv"))

    rel_df = pd.DataFrame()
    rel_df["id"] = ids
    rel_df["rels"] = rels
    rel_df.to_csv(os.path.join(out_dir, "relations.csv"))

    tk_df = pd.DataFrame()
    tk_df["id"] = ids
    tk_df['tokens'] = tokens
    tk_df.to_csv(os.path.join(out_dir, "tokens.csv"))


def process_CCHI():
    cchit = pd.read_csv("./data/CCHIT/artifacts.csv")
    stop_words = [
        x for x in pathlib.Path("./data/stop_words.txt").read_text().splitlines()
    ]
    out_dir = "./output/CCHIT/"
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    arts = cchit["arts"].to_list()
    ids = cchit["id"].to_list()

    dkg = DomainKG(stop_words=stop_words)
    concepts, rels, tokens = dkg.build(arts)
    concept_df = pd.DataFrame()
    concept_df["phrase"] = concepts
    concept_df["ids"] = ids
    concept_df.to_csv(os.path.join(out_dir, "concepts.csv"))

    rel_df = pd.DataFrame()
    rel_df["id"] = ids
    rel_df["rels"] = rels
    rel_df.to_csv(os.path.join(out_dir, "relations.csv"))

    tk_df = pd.DataFrame()
    tk_df["id"] = ids
    tk_df['tokens'] = tokens
    tk_df.to_csv(os.path.join(out_dir, "tokens.csv"))


if __name__ == "__main__":
    process_CM1()
    # process_CCHI()
    exit()
    # add more dataset here
    # test cases for concept extraction

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.getLogger("stanza").setLevel(logging.WARN)
    # concept_extraction_arts = [
    #     "The system shall use HL-7 communications protocol for transferring messages among packages",
    #     "The system shall allow event-delay capability for pre-admission, discharge, and transfer orders",
    # ]
    relation_extraction_arts = [
        "The system shall be able to support the standards identified and recommended by the Health Information Technology Standards Panel (HITSP) on its HITSP-TP13 Ver 1.0.1 document"
        "The system shall provide a way for user to quickly click on the specifications that he or she wants for each order placed",
        # "The system shall improve accessibility of online clinical information and results.",
        # "The system shall provide a platform for building interfaces to external lab services enabling automated order entry and results reporting.",
        "The system shall provide the ability to capture common content for prescription details including strength, sig, quantity, and refills to be selected by the ordering clinician.",
        # "The system shall have the ability to provide filtered displays of encounters based on encounter characteristics, including date of service, encounter provider and associated diagnosis.",
    ]
    dkg = DomainKG()
    concepts, relations = dkg.build(docs=relation_extraction_arts)
    for c in concepts:
        logger.debug(f"{c}")

    for r in relations:
        logger.debug(f"{r}")
