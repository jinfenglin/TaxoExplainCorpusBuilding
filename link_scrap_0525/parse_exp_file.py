import jsonlines

if __name__ == "__main__":
    with jsonlines.open("Explanation_0525.json") as fin, open("links.txt", "w",encoding="utf8") as fout:
        pairs = set()
        for item in fin:
            p, c, n = item["parent"], item["child"], item["len"]
            if n == 0:
                pairs.add((p, c))

        for p in pairs:
            fout.write(f"{p[0]},{p[1]}\n")
