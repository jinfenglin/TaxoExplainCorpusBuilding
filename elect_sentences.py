import os
from collections import defaultdict
from multiprocessing import Process, Queue
from pattmatch import kmp
from jsonlines import jsonlines

from EntityDetection import DomainKG


def write_results(results, out_dir):
    disc_definition = os.path.join(out_dir, "definition.jsonl")
    disc_context = os.path.join(out_dir, "context.jsonl")

    with jsonlines.open(disc_definition, "w") as fout:
        defs = results["definitions"]
        for cpt in defs:
            fout.write({"concept": cpt, "definition": list(defs[cpt])})
    with jsonlines.open(disc_context, "w") as fout:
        ctxs = results["context"]
        for cpt in ctxs:
            fout.write({"concept": cpt, "context": list(ctxs[cpt])})


def _worker_reduce(output_queue, map_num, out_dir):
    finished = 0
    definitions = defaultdict(set)
    context = defaultdict(set)

    while True:
        output = output_queue.get()
        if output is None:
            finished += 1
            if finished >= map_num:
                break
            else:
                continue

        if output["is_def"]:
            definitions[output["query"]].add(output["sent"])
        elif output["is_ctx"]:
            context[output["query"]].add(output["sent"])

    print("finished collecting results")
    bup_res = {"definitions": definitions, "context": context}
    write_results(bup_res, out_dir)


def _worker_map(job_queue, out_queue):
    dkg = DomainKG()
    while True:
        is_def, is_ctx = False, False
        job = job_queue.get()

        if job is None:
            break
        s, query = job['sent'], job['query']

        try:
            ann_sent = dkg.client.annotate(s).sentence[0]
        except:
            break
        if isinstance(query, list):
            c1, c2 = query[:2]
            query = f"{c1}|{c2}"
            t1 = is_definition(ann_sent, c1, s)
            t2 = is_definition(ann_sent, c2, s)
            type = None
            if t1 == 'def' or t2 == 'def':
                type = 'def'
            elif t1 == 'ctx' or t2 == 'ctx':
                type = 'ctx'
        else:
            type = is_definition(ann_sent, query, s)
        if type == "def":
            is_def = True
        elif type == "ctx":
            is_ctx = True

        out_queue.put({
            "query": query,
            "sent": s,
            "is_def": is_def,
            "is_ctx": is_ctx,
        })
    print("finished one process")
    out_queue.put(None)


def concurrent_get_definitions_and_related_concepts(clean_corpus, out_dir):
    map_num = 4
    mapworker = []
    job_q, out_q = Queue(), Queue()
    for i in range(4):
        w = Process(target=_worker_map, args=(
            job_q,
            out_q,
        ))
        mapworker.append(w)
        w.start()
    rp = Process(target=_worker_reduce, args=(
        out_q, map_num, out_dir
    ))
    rp.start()
    with jsonlines.open(clean_corpus) as fin:
        for obj in fin:
            sents = obj["sentences"]
            query = obj["query"]
            for s in sents:
                job_q.put({"sent": s, "query": query})
    for w in mapworker:
        job_q.put(None)
    for w in mapworker:
        w.join()
    rp.join()
    job_q.close()
    out_q.close()


def is_definition(asent, query, s):
    if s.endswith("?") or s.endswith('!') or query.lower() not in s.lower():
        return False
    query_idxs = kmp([x.word.lower() for x in asent.token], query.lower().split())
    if len(query_idxs) == 0:
        return None
    query_idxs = query_idxs[0]
    fidx, lidx = query_idxs[0], query_idxs[-1]
    pre_tks, post_tks = asent.token[:fidx], asent.token[lidx:]
    if len(post_tks) > 0:
        if fidx < 2:
            in_deps, out_deps = defaultdict(dict), defaultdict(dict)
            for r in asent.enhancedPlusPlusDependencies.edge:
                t1_idx, rel, t2_idx = r.source - 1, r.dep.split(":"), r.target - 1
                rel = rel[0]

                out_rels = out_deps[t1_idx]
                tmp = out_rels.get(rel, [])
                tmp.append(t2_idx)
                out_deps[t1_idx][rel] = tmp

                in_rels = in_deps[t2_idx]
                tmp = in_rels.get(rel, [])
                tmp.append(t1_idx)
                in_deps[t2_idx][rel] = tmp

            for idx in range(query_idxs[0], query_idxs[-1]):
                subj_dep = in_deps[idx].get("nsubj", [])
                if len(subj_dep) > 0:
                    obj = subj_dep[0]
                    if (
                            asent.token[obj].pos.startswith("NN")
                            and len(out_deps[obj].get("cop", [])) > 0
                    ):
                        cop_idx = out_deps[obj]['cop'][0]
                        if asent.token[cop_idx].word in ['is', 'are'] and (
                                fidx == 0 or asent.token[0].lemma in ['a', "an", "the"]):
                            return 'def'
                        else:
                            return 'ctx'
                    elif asent.token[obj].pos.startswith("VB") and asent.token[obj].pos not in ['VBD']:
                        return 'ctx'
        return None


if __name__ == "__main__":
    # clean_corpus = "./concept_corpus/concept_corpus.jsonl"
    # out_dir = f"./concept_corpus"
    # concurrent_get_definitions_and_related_concepts(clean_corpus, out_dir)

    clean_corpus = "./link_corpus/link_corpus.jsonl"
    out_dir = f"./link_corpus"
    concurrent_get_definitions_and_related_concepts(clean_corpus, out_dir)
