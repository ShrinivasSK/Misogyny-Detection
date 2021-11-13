# Core
import io  #open embedding file
import random  #fix random

# Basics
import pandas as pd
import numpy as np
import torch

# Utility
from tqdm import tqdm #progress-bar

# Dataloader
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# Model
from models import CNN_GRU_Model

# Optimiser
from transformers import AdamW

# Metrics
from sklearn.metrics import *

class CNN_GRU:
    def __init__(self,args):
        # fix the random
        random.seed(args['seed_val'])
        np.random.seed(args['seed_val'])
        torch.manual_seed(args['seed_val'])
        torch.cuda.manual_seed_all(args['seed_val'])
        
        # load embeddings
        self.vector,id2word,self.word2id = self.load_vec(args['embedding_path'])
        
        # set device
        self.device = torch.device(args['device'])

        self.model_save_path = args['model_save_path']
        self.name = args['name']
    
    ##-----------------------------------------------------------##
    ##------------------ Utility Functions ----------------------##
    ##-----------------------------------------------------------##
    def load_vec(self,emb_path, nmax=50000):
        vectors = []
        word2id = {}
        with io.open(emb_path, 'r', encoding='utf-8', newline='\n', 
                     errors='ignore') as f:
            next(f)
            for i, line in enumerate(f):
                # file is stored as <word> <array of floats>
                word, vect = line.rstrip().split(' ', 1)
                vect = np.fromstring(vect, sep=' ')
                assert word not in word2id, 'word found twice'
                vectors.append(vect)
                word2id[word] = len(word2id)
                if len(word2id) == nmax:
                    break
        id2word = {v: k for k, v in word2id.items()}
        embeddings = np.vstack(vectors)
        # add padding and unkown word embedding
        merged_vec = self.add_pad_unk(embeddings)
        return merged_vec, id2word, word2id
    
    def encode_data(self,data,max_len):
        new_data=[]
        
        for row in tqdm(data):
            encoded=[]
            words=row.split(' ')
            # as unknown is added after all words
            unk_index = len(list(self.word2id.keys()))
            # and padding after that
            pad_index = unk_index+1
            # to minimise extra loops, run only till max_len
            num = min(max_len,len(words))
            for word in words[0:num]:
                word=word.lower()
                try:
                    index=self.word2id[word]
                except KeyError:
                    # unkown word
                    index=unk_index
                encoded.append(index)
            if(len(encoded)<max_len):
                # add padding
                padding = [pad_index]*(max_len-len(encoded))
                encoded.extend(padding)
            else:
                # clip vector (although unneccessary here as 
                # we decreased loop range)
                encoded=encoded[0:max_len]
            new_data.append(encoded)
                                                   
        return new_data
    
    def add_pad_unk(self,vector):
        # padding is zeros vector
        pad_vec = np.zeros((1,vector.shape[1])) 
        # unkown is average of all
        unk_vec = np.mean(vector,axis=0,keepdims=True) 
        
        merged_vec=np.append(vector, unk_vec, axis=0)
        merged_vec=np.append(merged_vec, pad_vec, axis=0)
        
        return merged_vec
    
    ##-----------------------------------------------------------##
    ##------------------ Dataloader -----------------------------##
    ##-----------------------------------------------------------##
    def get_dataloader(self,X,Y, batch_size,max_len,is_train=False):
        # encode data
        inputs = self.encode_data(X,max_len)
        labels = Y

        # convert to tensors
        inputs = torch.tensor(inputs)
        labels = torch.tensor(labels,dtype=torch.long)

        # create dataset
        data = TensorDataset(inputs,labels)

        if(is_train):
            # use random sampler for training to shuffle
            # train data
            sampler = RandomSampler(data) 
        else:
            # order does not matter for validation as we just 
            # need the metrics
            sampler = SequentialSampler(data)

        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size,drop_last=True)

        return dataloader
    
    ##-----------------------------------------------------------##
    ##----------------- Training Utilities ----------------------##
    ##-----------------------------------------------------------##  
    
    def get_optimiser(self,learning_rate,model):
        # using AdamW optimiser from transformers library
        return AdamW(model.parameters(),
                  lr = learning_rate, 
                  eps = 1e-8
                )
        
    def evalMetric(self, y_true, y_pred, prefix):
        # calculate all the metrics and add prefix to them
        # before saving in dictionary
        accuracy = accuracy_score(y_true, y_pred)
        mf1Score = f1_score(y_true, y_pred, average='macro')
        f1Score = f1_score(y_true, y_pred)
        area_under_c = roc_auc_score(y_true, y_pred)
        recallScore = recall_score(y_true, y_pred)
        precisionScore = precision_score(y_true, y_pred)

        nonhate_f1Score = f1_score(y_true, y_pred, pos_label=0)
        non_recallScore = recall_score(y_true, y_pred, pos_label=0)
        non_precisionScore = precision_score(y_true, y_pred, pos_label=0)
        return {prefix+"accuracy": accuracy, prefix+'mF1Score': mf1Score, 
            prefix+'f1Score': f1Score, prefix+'auc': area_under_c,
            prefix+'precision': precisionScore, 
            prefix+'recall': recallScore, 
            prefix+'non_hatef1Score': nonhate_f1Score, 
            prefix+'non_recallScore': non_recallScore, 
            prefix+'non_precisionScore': non_precisionScore}
    
    ##-----------------------------------------------------------##
    ##---------------- Different Train Loops --------------------##
    ##-----------------------------------------------------------## 
    
    def evaluate(self,model,loader,which):
    
        model.eval() # put model in eval mode

        # maintain total loss to save in metrics
        total_eval_loss = 0

        # maintain predictions for each batch and calculate metrics
        # at the end of the epoch
        y_pred = np.zeros(shape=(0),dtype='int')
        y_true = np.empty(shape=(0),dtype='int')

        for batch in tqdm(loader):
            # separate input and labels
            b_inputs = batch[0].to(self.device)
            b_labels = batch[1].to(self.device)

            with torch.no_grad(): # do not construct compute graph
                outputs = model(b_inputs,b_labels)
            
            # output is always a tuple, thus we have to 
            # separate it manually
            loss = outputs[0]
            logits = outputs[1]

            # add the current loss
            # loss.item() extracts loss value as a float
            total_eval_loss += loss.item()

            # calculate true labels and convert it into numpy array
            b_y_true = b_labels.cpu().data.squeeze().numpy()

            # calculate predicted labels by taking max of 
            # prediction scores
            b_y_pred = torch.max(logits,1)[1]
            b_y_pred = b_y_pred.cpu().data.squeeze().numpy()

            y_pred = np.concatenate((y_pred,b_y_pred))
            y_true = np.concatenate((y_true,b_y_true))

        # calculate metrics
        metrics = self.evalMetric(y_true,y_pred,which+"_")

        # Calculate the average loss over all of the batches.
        avg_loss = total_eval_loss / len(loader)
        # add it to the metric
        metrics[which+'_avg_loss'] = avg_loss

        return metrics
    
    
    def run_train_loop(self,model,train_loader,optimiser):
        # to evaluate model on test and validation set
        
        model.train() # put model in train mode

        # maintain total loss to add to metric
        total_loss = 0

        # maintain predictions for each batch and calculate metrics
        # at the end of the epoch
        y_pred = np.zeros(shape=(0),dtype='int')
        y_true = np.empty(shape=(0),dtype='int')

        for batch in tqdm(train_loader):
            # separate inputs and labels
            b_inputs = batch[0].to(self.device)
            b_labels = batch[1].to(self.device)
            
#             print(b_inputs.shape,b_labels.shape)

            # Ref: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch#:~:text=In%20PyTorch%20%2C%20we%20need%20to,backward()%20call.
            model.zero_grad()        

            outputs = model(b_inputs,b_labels)
            
            # outputs is always returned as tuple
            # Separate it manually
            loss = outputs[0]
            logits = outputs[1]

            # calculate current loss
            # loss.item() extracts loss value as a float
            total_loss += loss.item()

            # Back-propagation
            loss.backward()

            # calculate true labels
            b_y_true = b_labels.cpu().data.squeeze().numpy()

            # calculate predicted labels by taking max of 
            # prediction scores
            b_y_pred = torch.max(logits,1)[1]
            b_y_pred = b_y_pred.cpu().data.squeeze().numpy()

            y_pred = np.concatenate((y_pred,b_y_pred))
            y_true = np.concatenate((y_true,b_y_true))

            # clip gradient to prevent exploding gradient
            # problems
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # gradient descent
            optimiser.step()

        # calculate avg loss 
        avg_train_loss = total_loss / len(train_loader)

        # calculate metrics
        train_metrics = self.evalMetric(y_true,y_pred,"Train_")

        # print results
        print('avg_train_loss',avg_train_loss)
        print('train_f1Score',train_metrics['Train_f1Score'])
        print('train_accuracy',train_metrics['Train_accuracy'])
        
        # add loss to metrics
        train_metrics['Train_avg_loss'] = avg_train_loss

        return train_metrics
    
    
    ##------------------------------------------------------------##
    ##----------------- Main Train Loop --------------------------##
    ##------------------------------------------------------------##
    
    def train(self,model,data_loaders,optimiser,epochs,save_model):
        # save train stats per epoch
        train_stats = []
        train_loader,val_loader,test_loader = data_loaders
        # maintain best mF1 Score to save best model
        best_mf1Score=-1.0
        for epoch_i in range(0, epochs):
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            
            print("")
            print('Training...')
            # run trian loop
            train_metrics = self.run_train_loop(model,train_loader,optimiser)

            print("")
            print("Running Validation...") 
            # test on validation set
            val_metrics = self.evaluate(model,val_loader,"Val")
            
            print("Validation Loss: ",val_metrics['Val_avg_loss'])
            print("Validation Accuracy: ",val_metrics['Val_accuracy'])
            
            stats = {}
            # save model where validation mF1Score is best
            if(val_metrics['Val_mF1Score']>best_mf1Score):
                print("Testing....")
                best_mf1Score=val_metrics['Val_mF1Score']
                if(save_model):
                    torch.save(model.state_dict(), self.model_save_path+
                            '/best_cnn_gru_'+self.name+'.pt')
                # evaluate best model on test set
                test_metrics = self.evaluate(model,test_loader,"Test")

            stats['epoch']=epoch_i+1

            # add train and val metrics of the epoch to 
            # same dictionary
            stats.update(train_metrics)
            stats.update(val_metrics)

            train_stats.append(stats)

        return train_stats,test_metrics
    
    ##-----------------------------------------------------------##
    ##------------------------ The Pipeline ---------------------##
    ##-----------------------------------------------------------##
    def run(self,args,df_train,df_val,df_test):
        # get X and Y data points 
        X_train = df_train['Text'].values
        Y_train = df_train['Label'].values
        X_test = df_test['Text'].values
        Y_test = df_test['Label'].values
        X_val = df_val['Text'].values
        Y_val = df_val['Label'].values

        # convert them to data loaders
        # encoding input done here
        train_dl = self.get_dataloader(X_train,Y_train,
                        args['batch_size'],args['max_len'],True)
        val_dl = self.get_dataloader(X_val,Y_val,args['batch_size'],
                               args['max_len'])
        test_dl = self.get_dataloader(X_test,Y_test,args['batch_size'],
                                args['max_len'])
        
        # initialise model
        model = CNN_GRU_Model(args['model'],self.vector,args['device'])
        
        model.to(self.device)
        
        optimiser=self.get_optimiser(args['learning_rate'],model)
        
        # Run train loop and evaluate on validation data set
        # on each epoch. Store best model from all epochs 
        # (best mF1 Score on Val set) and evaluate it on
        # test set
        train_stats,test_metrics = self.train(model,[train_dl,val_dl,test_dl],
                            optimiser,args['epochs'],args['save_model'])
        return train_stats,test_metrics
    
    ##-----------------------------------------------------------##
    ##-------------------- Other Utilities ----------------------##
    ##-----------------------------------------------------------##
    def run_test(self,model,df_test,args):
        # to evaluate test set on the final saved model
        # to retrieve results if necessary
        X_test = df_test['Text'].values
        Y_test = df_test['Label'].values

        test_dl = self.get_dataloader(X_test,Y_test,32,
                                args['max_len'])

        metrics = self.evaluate(model,test_dl,"Test")
        return metrics
    
    def load_model(self,path,args):
        # load saved best model
        saved_model=CNN_GRU_Model(args['model'],self.vector)
        
        saved_model.load_state_dict(torch.load(path))
        
        return saved_model