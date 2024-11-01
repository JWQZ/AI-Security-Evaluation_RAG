from langchain_community.document_loaders import *
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from datasets import load_from_disk
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import json
import transformers
from os import path
from tqdm import tqdm
import ast

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

BASE_DATA_PATH="/mnt/data/chenjinwen"

# print(torch.cuda.device_count())
# print(torch.cuda.current_device())
device=torch.device("cuda:0")
# 加载文档
def load_documents(directory='/mnt/data/chenjinwen/AI-Security-Evaluation_RAG/data/wiki.train.txt'):
    if "popQA" in directory:
        text_splitter=RecursiveCharacterTextSplitter(separators=["\n"])
        with open(directory, 'r', encoding='utf-8') as file:
            texts=file.readlines()
        decuments=text_splitter.create_documents(texts=texts)   
        return decuments
    loader = TextLoader(directory)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=30)
    text_splitter=RecursiveCharacterTextSplitter(separators="\n")
    split_docs = text_splitter.split_documents(documents)
    return split_docs


# 加载embedding模型
def load_embedding_model(model_path='/mnt/data/chenjinwen/AI-Security-Evaluation_RAG/models/bge-large-en-v1.5'):
    embedding_model = HuggingFaceEmbeddings(  # embedding 模型
        model_name=model_path,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    return embedding_model


def load_llama(model_path='/mnt/data/chenjinwen/AI-Security-Evaluation_RAG/models/Llama-2-7b-chat-hf'):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map={"": device},
        low_cpu_mem_usage=True
    )
    model = base_model.eval()
    return model, tokenizer


def load_qa(dataset_path='/mnt/data/chenjinwen/AI-Security-Evaluation_RAG/data/NQ-dev.json'):
    if "PopQA" in dataset_path:
        dataset=load_from_disk(dataset_path)
        return dataset['question'],dataset['possible_answers'],dataset
    question = []
    answer = []
    with open(dataset_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                # 使用json.loads()解析每个JSON对象
                data = json.loads(line)
                # 处理JSON数据
                question.append(data['question'])
                answer.append((data['answer']))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
            except Exception as e:
                print(f"An error occurred: {e}")
    return question, answer
def load_Qwen(model_path="/mnt/data/chenjinwen/AI-Security-Evaluation_RAG/models/Qwen2.5-7B-Instruct"):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map={'':device}
    )
    model=model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer
def get_answer_Qwen(question, model, tokenizer, vector_store, device, k=6, rag=False):
    if rag:
        docs=vector_store.similarity_search(question, k=k)  # 计算相似度，并把相似度高的chunk放在前面
        contexts = [doc.page_content for doc in docs]
        contexts_text="\n".join(contexts)       
        system_prompt = f"You are now playing the role of the encyclopedia. Answer my questions without any unnecessary words. Here is some knowledge you can refer to:\n{contexts_text}"   
    else:
        system_prompt = "You are now playing the role of the encyclopedia. Answer my questions without any unnecessary words."
    if not question.endswith("?"):
        question = question + "?"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    if len(model_inputs.input_ids[0]) > 30000:
        return get_answer_Qwen(question, model, tokenizer, vector_store, device, k=k-1, rag=rag)
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    if rag:
        return response,contexts
    else:
        return response
    
def get_answer_llama2(question, model, tokenizer, vector_store, device, k=6, rag=False):
    if not question.endswith("?"):
        question = question + "?"
    if rag and k>0:
        docs=vector_store.similarity_search(question, k=6)  # 计算相似度，并把相似度高的chunk放在前面
        contexts = [doc.page_content for doc in docs]
        contexts_text="\n".join(contexts)
        input=f"<s>[INST] <<SYS>>\nYou are now playing the role of the encyclopedia. Answer my questions without any unnecessary words.\nYou can use the following knowledge but don't be misled by inaccurate content:\n{contexts_text}\n<</SYS>>nQuestion:\n{question}\nAnswer: [/INST]"       
    else:
        input = f"<s>[INST] <<SYS>>\nYou are now playing the role of the encyclopedia. Answer my questions without any unnecessary words.\n<</SYS>>\nQuestion:\n{question}\nAnswer: [/INST]"
    inputs = tokenizer(input, return_tensors='pt', max_length=4000, truncation=True)
    if inputs["input_ids"].shape[1] > 3900:
        return get_answer_llama2(question, model, tokenizer, vector_store, device, k=k-1, rag=rag)
    with torch.no_grad():
        generate_ids = model.generate(input_ids=inputs["input_ids"].to(device),max_new_tokens=50)
    ans_text=tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    ans_text=ans_text.split("[/INST]")[1].strip()
    if rag:
        if k<=0:
            return ans_text,[]
        return ans_text,contexts
    else:
        return ans_text
def inference(qa_dataset_path="/mnt/data/chenjinwen/AI-Security-Evaluation_RAG/data/PopQA"):
    embedding_model = load_embedding_model()
    knowledge_dir='/mnt/data/chenjinwen/AI-Security-Evaluation_RAG/data/popQA_knowledge'
    knowledge_file_dir='/mnt/data/chenjinwen/AI-Security-Evaluation_RAG/data/popQA_knowledge.txt'
    if not os.path.exists(knowledge_dir):
        documents = load_documents(knowledge_file_dir)
        vector_store = FAISS.from_documents(documents, embedding_model)
        vector_store.save_local(knowledge_dir)
    else:
        # 如果本地已经有faiss仓库了，说明之前已经保存过了，就直接读取
        vector_store = FAISS.load_local(knowledge_dir, embeddings=embedding_model, allow_dangerous_deserialization=True)
    questions, answers, dataset = load_qa(qa_dataset_path)
    # model, tokenizer = load_Qwen("/mnt/data/chenjinwen/AI-Security-Evaluation_RAG/models/Qwen2.5-7B-Instruct")
    model, tokenizer = load_llama("/mnt/data/chenjinwen/AI-Security-Evaluation_RAG/models/Llama-2-7b-chat-hf")
    answers_plain=[]
    answers_rag=[]
    contexts_list=[]
    progass_bar=tqdm(range(len(questions)))
    with torch.no_grad():
        for question in questions:
            # answer_plain = get_answer_Qwen(question, model, tokenizer, vector_store, device)
            # answer_rag,contexts = get_answer_Qwen(question, model, tokenizer, vector_store, device,rag=True)
            answer_plain = get_answer_llama2(question, model, tokenizer, vector_store, device,rag=False)
            answer_rag,contexts = get_answer_llama2(question, model, tokenizer, vector_store, device,rag=True)
            answers_plain.append(answer_plain.strip())
            answers_rag.append(answer_rag.strip())
            contexts_list.append(contexts)
            progass_bar.update(1)
    def map_func(data,idx):
        data['answer_plain']=answers_plain[idx]
        data['answer_rag']=answers_rag[idx]
        data['context_rag']=contexts_list[idx]
        return data
    dataset=dataset.map(map_func,with_indices=True)
    dataset.save_to_disk(qa_dataset_path+"_output")    
    progass_bar.close()
def cal_metrics(qa_out_dataset_path):
    dataset_out = load_from_disk(qa_out_dataset_path)
    answers_plain = dataset_out['answer_plain']
    answers_rag = dataset_out['answer_rag']
    possible_answers = dataset_out['possible_answers']
    count_all = 0
    count_acc_plain = 0
    count_acc_rag = 0
    for answer_plain, answer_rag, possible_answer in zip(answers_plain, answers_rag, possible_answers):
        count_all += 1
        # print(possible_answer)
        for ans in ast.literal_eval(possible_answer):
            if ans.lower() in answer_plain.lower():
                # print(f"Plain: {ans}")
                # print(f"Plain: {answer_plain}")

                count_acc_plain += 1
                break
        for ans in ast.literal_eval(possible_answer):
            if ans.lower() in answer_rag.lower():
                count_acc_rag += 1
                break
    acc_plain = count_acc_plain / count_all
    acc_rag = count_acc_rag / count_all
    print(f"Plain accuracy: {acc_plain}")
    print(f"Rag accuracy: {acc_rag}")
if __name__ == '__main__':
    # print(torch.cuda.current_device())
    # print(torch.cuda.is_available())
    # inference(qa_dataset_path="/mnt/data/chenjinwen/AI-Security-Evaluation_RAG/data/PopQA_100_llama2")
    cal_metrics("/mnt/data/chenjinwen/AI-Security-Evaluation_RAG/data/PopQA_5000_output")