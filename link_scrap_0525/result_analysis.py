import jsonlines

if __name__ == "__main__":
    has_sent, sent_num = 0, 0
    with jsonlines.open("link_corpus.jsonl") as fin:
        pairs = set()
        for item in fin:
            n = int(item["sent_num"])
            if n > 0:
               has_sent += 1
               sent_num += n
    print(f"{has_sent}, avg_len = {sent_num/has_sent}")

