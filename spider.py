import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from datasets import load_from_disk
import time
from concurrent.futures import ThreadPoolExecutor
def url2knowledge(url):
    # 发送GET请求
    # print(url)
    headers={
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36'
    }
    response = requests.get(url,headers=headers)
    knowledge=''
    # 检查请求是否成功
    if response.status_code == 200:
        # 使用BeautifulSoup解析HTML内容
        # print(response.text)
        soup = BeautifulSoup(response.text, 'html.parser')
        # print(soup)
        titles=soup.find_all('div',class_='wikibase-statementgroupview-property')
        tables=soup.find_all('div',class_='wikibase-statementlistview')
        try:
            assert len(titles)==len(tables)
            # print(url)
        except AssertionError as e:
            print(e)
        head=soup.find('span',class_='wikibase-title-label')
        head_description=soup.find('div',class_='wikibase-entitytermsview-heading-description')
        aliases=soup.find_all('li',class_='wikibase-entitytermsview-aliases-alias')
        if head!=None:
            knowledge+=head.get_text()
            knowledge+=': '
        if head_description!=None:
            knowledge+=head_description.get_text()
            knowledge+=', '
        if aliases!=[]:
            knowledge+="alias: "
            knowledge+=str([alias.get_text() for alias in aliases])
        # print(head.get_text(),',',head_description.get_text(),',',[alias.get_text() for alias in aliases])

        # print(len(tables))
        # print(len(titles))
        # print(tables[0])
        for title,table in zip(titles,tables):
            # bodys=table.find_all('div',class_='wikibase-snakview-body')
            bodys=table.find_all('div',class_='wikibase-statementview-mainsnak-container')
            knowledge+=", "
            knowledge+=title.get_text().strip()
            knowledge+=": "
            knowledge+=str([x.find('div',class_='wikibase-statementview-mainsnak').get_text().strip() for x in bodys])
            # print(title.get_text().strip(),[x.get_text().strip() for x in bodys])
            # print("------------------")
        return knowledge
    else:
        print('请求失败，状态码：', response.status_code)
        return url
def urls2txt(urls,data_path):
    knowledges=[]
    with open(data_path,'r',encoding='utf-8') as f:
        len_now=len(f.readlines())
    for i,url in enumerate(urls):
        if i<len_now:
            continue
        knowledge=url2knowledge(url.replace("entity","wiki"))
        if knowledge==url.replace("entity","wiki"):
            print(knowledge)
            continue
        with open(data_path,'+a',encoding='utf-8') as f:
            try:
                f.write(knowledge)
                f.write("\n")
            except UnicodeEncodeError as e:
                print(url)
                print(knowledge)
                print(e)

        
if __name__ == '__main__':
    dataset=load_from_disk("D:\BaiduSyncdisk\AI-Security-Evaluation_RAG\data\PopQA")
    urls=dataset['s_uri']
    data_path="D:\BaiduSyncdisk\AI-Security-Evaluation_RAG\data\popQA_knowledge.txt"
    # prograss_bar=tqdm(total=len(dataset['s_uri']))
    # for s_uri in dataset['s_uri']:
    #     prograss_bar.update(1)
    #     knowledge=url2knowledge(s_uri.replace("entity","wiki"))
    #     if knowledge==s_uri:
    #         print(knowledge)
    #         continue
    #     with open("D:\BaiduSyncdisk\AI-Security-Evaluation_RAG\data\popQA_knowledge.txt",'+a',encoding='utf-8') as f:
    #         try:
    #             f.write(knowledge)
    #             f.write("\n")
    #         except UnicodeEncodeError as e:
    #             print(s_uri)
    #             print(knowledge)
    #             print(e)
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures=[]
        for i in range(9):
            futures.append(executor.submit(urls2txt,urls[i*1400:(i+1)*1400],data_path.replace(".txt",f"_{i}.txt")))
        futures.append(executor.submit(urls2txt,urls[12600:],data_path.replace(".txt","_9.txt")))
        for future in futures:
            future.result()
