from conllu import parse
import os, re
def convertUD(d_name = "UD_Japanese-GSD"):
    p = re.compile('.*\.conllu')
    q = re.compile('.*dev.*')

    fs = []
    for file in os.listdir(d_name):
        if p.match(file):
            if not q.match(file):
                fs.append(file)

    txt = ""
    for f in fs:
        o = open(d_name + "/" + f, "r")
        txt += o.read()
        o.close()

    sentences = parse(txt)

    p_sentences = []
    for s in sentences:
        temp = []
        for w in s:
            temp.append((w['form'], w['upos']))
        p_sentences.append(temp)
    return p_sentences

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", help="path to the Universal Dependencies data directory", default="UD_Japanese-GSD")
    args = parser.parse_args()
    sentences = convertUD(args.directory)
    print(len(sentences))