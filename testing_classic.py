from data_cleaning import Data_Preprocessing
from CNN_GRU import CNN_GRU
from LSTM import LSTM
from bert import BERT
from classic_models import Classic
import pandas as pd
import numpy as np
import torch
import random
from tqdm import tqdm

from gensim.models import FastText

from nltk.corpus import stopwords

# fix the random
def fix_random(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

# load dataset 
# keep count of hate and non_hate data
# equal to train_cnt in train set 
def load_dataset(args,index,run):
    # initialise constants 
    path = args['data_path']
    train_cnt = args['train_cnt']
    model_name = args['model_name']
    # read dataframes
    df_train = pd.read_csv(path+'train_'+str(index)+'.csv')
    df_val = pd.read_csv(path+'val_'+str(index)+'.csv')
    df_test = pd.read_csv(path+'test_'+str(index)+'.csv')
    
    # split train into hate and non-hate and take train_cnt
    # samples of each
    df_train_hate = df_train[df_train['Label'] == 1].sample(train_cnt,random_state=seeds[run-1])
    df_train_non_hate = df_train[df_train['Label'] == 0].sample(train_cnt,random_state=seeds[run-1])
    # concatenate hate and non_hate
    df_train = pd.concat([df_train_hate, df_train_non_hate])
    # shuffle the train data
    df_train = df_train.sample(frac=1).reset_index(drop=True)

    # clean data
    df_train=clean_data(df_train,model_name)
    df_val=clean_data(df_val,model_name)
    df_test=clean_data(df_test,model_name)

    return df_train, df_val, df_test


def clean_data(df,model_name):
    X = df['Text']
    processer = Data_Preprocessing('es_core_news_sm')
    X_new=[]
    if(model_name!='bert'):
        stop_words = set(stopwords.words('spanish'))
        for text in tqdm(X):
            text= processer.removeEmojis(text)
            text = processer.removeUrls(text)
            text= processer.removeSpecialChar(text)
            text = processer.removeStopWords(text,stop_words)
            text= processer.lemmatise(text)
            X_new.append(text)
    else:
        for text in tqdm(X):
            text= processer.removeEmojis(text)
            text = processer.removeUrls(text)
            text=processer.removeSpecialChar(text)
            X_new.append(text)
    df['Text']=X_new
    return df 

# Choose model
def choose_model(name, index,run, train_cnt,run_args):
    args = argsAll[name]

    args['name'] = str(index)+'_'+str(train_cnt)+'_'+str(run)
    args['model_save_path']=run_args['model_save_path']

    if(name == 'lstm'):
        model = LSTM(args)
    elif(name == 'cnn_gru'):
        model = CNN_GRU(args)
    elif(name == 'bert'):
        model = BERT(args)
    elif(name=='classic'):
        model=Classic(args)

    return model, args

def save_metrics(path,metrics,which):
    df = pd.DataFrame(metrics)
    df.to_csv(path+"_"+which+".csv")


def train(args, index,run,all_test_metrics):
    model_name = args['model_name']
    print("\tInitialising Model....")
    model, model_args = choose_model(model_name,index,run,args['train_cnt'],args)
    print("\tLoading Dataset....")
    df_train, df_val, df_test = load_dataset(args,index,run)
    print("\tTraining Starts....")
    if(model_name=='classic'):
        test_metrics = model.run(model_args,df_train,df_test)
        print("Saving Test Metrics....")
        save_metrics(args['res_base_path']+args['model_name']+
             '_'+str(args['train_cnt'])+'_'+str(index)+'_'+str(run),test_metrics,"test")
    else:
        train_metrics, test_metrics = model.run(model_args, 
                        df_train, df_val, df_test)
        
        # Save train metrics after generating path
        res_path=args['res_base_path']+model_name+'_'+model_args['name']
        save_metrics(res_path,train_metrics,"train")
    
        test_metrics['name']=model_args['name']
    
        all_test_metrics.append(test_metrics)

sizes = [32, 64, 128, 256, 512]
models = ['lstm', 'cnn_gru', 'bert','classic']
folds = [1, 2, 3, 4, 5]
seeds = [43,44,45]

def run(args):
    all_test_metrics=[]
    
    for fold in folds:
        print("Fold: ",fold)
        # run thrice for each fold
        for i in range(0,3):
            fix_random(seeds[i])
            print("Run: ",i+1)
            train(args,fold,i+1,all_test_metrics)


# res_path = "Embeddings/hin_codemixed.model"
# model_1 = FastText.load(res_path)
            
argsAll = {
    'lstm': {
        'seed_val': 42,
        'batch_size': 32,
        'max_len': 1024,
        'weights': [1.0, 1.0],
        'epochs': 10,
        'learning_rate': 1e-4,
        'device': 'cpu',
        'embedding_path': "Embeddings/cc.en.300.vec",
        'save_model': False,
    },
    'bert': {
        'seed_val': 42,
        'batch_size': 32,
        'bert_model': "bert-base-multilingual-cased",
        'learning_rate': 2e-5,
        'epochs': 4,
        'max_len': 256,
        'device': 'cpu',
        'weights': [1.0, 1.0],
        'save_model': False,
    },
    'cnn_gru': {
        'seed_val': 42,
        'embedding_path': "Embeddings/cc.en.300.vec",
        'batch_size': 32,
        'learning_rate': 1e-4,
        'epochs': 10,
        'device': 'cpu',
        'max_len': 1024,
        'model': {
            'train_embed': False,
            'weights': [1.0, 1.0],
        },
        'save_model': False,
    },
    'classic':{
        'embedding': 'w2v',
        'embedding_path': "Embeddings/cc.en.300.vec",
        'max_len': 128,
        'top_k': 20000,
        'lang':'es',
        'seed_val':42,
        'weights': {0:1,1:1},
        'save_model': False,
#         'model': model_1,
    }
}

embeddings = ['bow','ngram','tfidf','fasttext','laser']

run_args={
    'model_name':'classic',
    'data_path':'Data_Processed/AMI-Spanish/',
    'train_cnt':32,
    'res_path': 'Results/AMI-Spanish/Classic_laser/',
    'model_save_path': 'Saved_Models/AMI-Spanish/',
}


# print("yo")

# for embedding in embeddings:
embedding='laser'
print("Embedding: ",embedding)
print('Data: ',run_args['data_path'].split('/')[-2])
argsAll['classic']['embedding']=embedding
run_args['res_base_path']=run_args['res_path']

for cnt in sizes:
    print("Train Cnt: ",cnt)
    run_args['train_cnt']=cnt
    run(run_args)
        

# for cnt in sizes[3:]:
#     print("Train Cnt: ",cnt)
#     run_args['train_cnt']=cnt
#     run(run_args)