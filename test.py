import gzip,json

if __name__ == '__main__':
    with gzip.open("./data/latest-lexemes.json.gz") as f:
        datas=json.load(f)
        # print(datas)
    with open("./data/latest-lexemes.json","w") as f:
        json.dump(datas,f)
