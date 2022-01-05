# Core
import random

# Basics
import numpy as np
import pandas as pd
import torch

# Metrics
from sklearn.metrics import *

# Tokeniser
from transformers import RobertaTokenizer

# Utility
from tqdm import tqdm

# Dataloader
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# Scheduler
from transformers import get_linear_schedule_with_warmup

# Optimiser
from transformers import AdamW

# Model

import torch.nn as nn
from models import weighted_Roberta


class XLM_Roberta:
    def __init__(self,args):
        # fix the random
        random.seed(args['seed_val'])
        np.random.seed(args['seed_val'])
        torch.manual_seed(args['seed_val'])
        torch.cuda.manual_seed_all(args['seed_val'])
        
        # set device
        self.device = torch.device(args['device'])

        self.weights=args['weights']
        
        # initiliase tokeniser
        self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base', do_lower_case = True)

        self.model_save_path = args['model_save_path']
        self.name = args['name']
        
    ##-----------------------------------------------------------##
    ##----------------- Utility Functions -----------------------##
    ##-----------------------------------------------------------##
    def encode(self,data,max_len):
        input_ids = []
        attention_masks = []
        for sent in tqdm(data):
            # use in-built tokeniser of Bert
            encoded_dict = self.tokenizer.encode_plus(
                            sent,
                            add_special_tokens =True, # for [CLS] and [SEP]
                            max_length = max_len,
                            truncation = True,
                            padding = 'max_length',
                            return_attention_mask = True,
#                             return_tensors = 'pt', # return pytorch tensors
            )
            input_ids.append(encoded_dict['input_ids'])
            # attention masks notify where padding has been added 
            # and where is the sentence
            attention_masks.append(encoded_dict['attention_mask'])
            X_data = torch.tensor(input_ids)
            attention_masks_data = torch.tensor(attention_masks)
            
        return [X_data,attention_masks_data]
    
    ##-----------------------------------------------------------##
    ##------------------ Dataloader -----------------------------##
    ##-----------------------------------------------------------##
    def get_dataloader(self,samples, batch_size,is_train=False):
        inputs,masks,labels = samples

        # Convert the lists into tensors.
        inputs = torch.cat(inputs, dim=0)
        masks = torch.cat(masks, dim=0)
        labels = torch.tensor(labels)

        # convert to dataset
        data = TensorDataset(inputs,masks,labels)

        if(is_train==False):
            # use random sampler for training to shuffle
            # train data
            sampler = SequentialSampler(data)
        else:
            # order does not matter for validation as we just 
            # need the metrics
            sampler = RandomSampler(data)  

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
    
    def get_scheduler(self,epochs,optimiser,train_dl):
        total_steps = len(train_dl) * epochs
        return get_linear_schedule_with_warmup(optimiser, 
                num_warmup_steps = 0, 
                num_training_steps = total_steps)
    
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
        # to evaluate model on test and validation set

        model.eval() # put model in eval mode

        # maintain total loss to save in metrics
        total_eval_loss = 0

        # maintain predictions for each batch and calculate metrics
        # at the end of the epoch
        y_pred = np.zeros(shape=(0),dtype='int')
        y_true = np.empty(shape=(0),dtype='int')

        for batch in tqdm(loader):
            # separate input, labels and attention mask
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)

            with torch.no_grad(): # do not construct compute graph
                outputs = model(b_input_ids, 
                                   token_type_ids=None, 
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
            
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
    
    
    def run_train_loop(self,model,train_loader,optimiser,scheduler):

        model.train() # put model in train mode

        # maintain total loss to add to metric
        total_loss = 0

        # maintain predictions for each batch and calculate metrics
        # at the end of the epoch
        y_pred = np.zeros(shape=(0),dtype='int')
        y_true = np.empty(shape=(0),dtype='int')

        for batch in tqdm(train_loader):
            # separate inputs, labels and attention mask
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)

            # Ref: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch#:~:text=In%20PyTorch%20%2C%20we%20need%20to,backward()%20call.
            model.zero_grad()                

            outputs = model(b_input_ids, 
                             token_type_ids=None, 
                             attention_mask=b_input_mask, 
                             labels=b_labels)

            # outputs is always returned as tuple
            # Separate it manually
            logits = outputs[0]

            # define new loss function so that we can include
            # weights
            loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(
                        self.weights,dtype=torch.float).to(self.device))
            
            loss = loss_fct(logits,b_labels)
            
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
            
            # schedule learning rate accordingly
            scheduler.step()

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
    def train(self,model,data_loaders,optimiser,scheduler,epochs,save_model):
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
            train_metrics = self.run_train_loop(model,train_loader,
                                            optimiser,scheduler)

            print("")
            print("Running Validation...") 
            # test on validation set
            val_metrics = self.evaluate(model,val_loader,"Val")
            
            print("Validation Loss: ",val_metrics['Val_avg_loss'])
            print("Validation Accuracy: ",val_metrics['Val_accuracy'])
            
            stats = {}

            # save model where validation mF1Score is best
            if(val_metrics['Val_mF1Score']>best_mf1Score):
                best_mf1Score=val_metrics['Val_mF1Score']
                if(save_model):
                    torch.save(model.state_dict(), self.model_save_path+
                        '/best_bert_'+self.name+'.pt')
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
    ##----------------------- Main Pipeline ---------------------##
    ##-----------------------------------------------------------##
    def run(self,args,df_train,df_val,df_test):
        # get X and Y data points 
        X_train = df_train['Text'].values
        Y_train = df_train['Label'].values
        X_test = df_test['Text'].values
        Y_test = df_test['Label'].values
        X_val = df_val['Text'].values
        Y_val = df_val['Label'].values
        
        # encode data
        # returns list of data and attention masks
        train_data = self.encode(X_train,args['max_len'])
        val_data = self.encode(X_val,args['max_len'])
        test_data = self.encode(X_test,args['max_len'])
        
        # add labels to data so that we can send them to
        # dataloader function together
        train_data.append(Y_train)
        val_data.append(Y_val)
        test_data.append(Y_test)
        
        # convert to dataloader
        train_dl =self.get_dataloader(train_data,args['batch_size'],True)
        val_dl =self.get_dataloader(val_data,args['batch_size'])                          
        test_dl =self.get_dataloader(test_data,args['batch_size'])
        
        # intialise model
        model = weighted_Roberta.from_pretrained(
            'xlm-roberta-base', # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = 2, # The number of output labels--2 for binary classification             # You can increase this for multi-class tasks.   
            params=args['params'],
        )
        model.to(self.device)
        
        optimiser = self.get_optimiser(args['learning_rate'],model)
        
        scheduler = self.get_scheduler(args['epochs'],optimiser,train_dl)
        
        # Run train loop and evaluate on validation data set
        # on each epoch. Store best model from all epochs 
        # (best mF1 Score on Val set) and evaluate it on
        # test set
        train_stats,train_metrics = self.train(model,[train_dl,val_dl,test_dl],
                                optimiser,scheduler,args['epochs'],args['save_model'])
        
        return train_stats,train_metrics
        
    ##-----------------------------------------------------------##
    ##-------------------- Other Utilities ----------------------##
    ##-----------------------------------------------------------##
    def run_test(self,model,df_test,args):
        # to evaluate test set on the final saved model
        # to retrieve results if necessary
        X_test = df_test['Text'].values
        Y_test = df_test['Label'].values

        test_data = self.encode(X_test,args['max_len'])

        test_data.append(Y_test)

        test_dl =self.get_dataloader(test_data,32)

        metrics = self.evaluate(model,test_dl,"Test")

        return metrics
    
    def load_model(self,path,args):
        # load saved best model
        model = XLM_RobertaModel.from_pretrained(
            args['bert_model'], 
            num_labels = 2, # The number of output labels--2 for binary classification                
            args=args,
        )
        
        saved_model.load_state_dict(torch.load(path))
        
        return saved_model