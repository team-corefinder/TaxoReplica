import argparse
import os
import torch
import gensim
import dgl
import dgl.function as fn
import torch.optim as optim
import math
from torch.utils.data import TensorDataset 
from torch.utils.data import DataLoader
from torch import nn
from encoder import DocuEncoder, ClassEncoder
from layer import GCN
from classifier import TextClassifier
from preprocessor import TaxoDataManager, DocumentManager
from gensim.test.utils import datapath
from gensim.models import word2vec



if __name__ == '__main__':
        if torch.cuda.is_available():
                print("CUDA is available!")
        else :
                print("CUDA is unavailable.")


        root = '/root/text-classifier/TaxoReplica/text-classifier/'
        train_file = 'train-with-core-class-1000.jsonl'
        taxonomy_file = 'taxonomy.json'
        data_name = 'amazon'

        d_encoder = DocuEncoder()

        word2vec_model = word2vec.Word2Vec.load(root + 'pretrained/' + 'embedding')
        W = word2vec_model.wv.vector_size

        gcn_model = GCN(W, W, W, 2, nn.ReLU())

        class_encoder = ClassEncoder(gcn_model, word2vec_model)



        


        tm = TaxoDataManager(root + 'processed_data/', taxonomy_file, data_name, word2vec_model)
        tm.load_all()
        dm = DocumentManager(train_file, root + 'processed_data/', 'amazon_train', d_encoder.Tokenize,  tm)
        dm.load_tokens()
        dm.load_dicts()

        #token length
        T = 512
        g = tm.get_graph().to('cuda:0')
        L = len(g.nodes())

        features = tm.get_feature().cuda()

        text_classifier = TextClassifier(class_encoder, d_encoder, (W, 768), T, g, features)

        sum = 0
        for c in gcn_model.parameters():
                sum = sum + 1
        print("GCN total Layer: ", sum)



        sum = 0
        for c in text_classifier.parameters():
                sum = sum + 1
        print("Text-classfier Layer: ", sum)

        loss_fun = torch.nn.BCELoss(reduction='sum')
        optimizer = optim.AdamW([
                {'params': text_classifier.document_encoder.parameters(), 'lr': 0.00005},
                {'params': text_classifier.class_encoder.parameters()},
                {'params': text_classifier.weight}], lr=0.004)



        stdv = 1. / math.sqrt(W)

        for i, row in enumerate(features,0):
                features[i] = row.data.uniform_(-stdv, stdv)
        
        train_ids = dm.get_ids()



        for i, document_id in enumerate(train_ids, 0):
                tokens = torch.tensor( dm.get_tokens(document_id) ,dtype = torch.int32)
                tokens = torch.reshape(tokens, (-1, 1))
                pos, nonneg = dm.get_output_label(document_id)
                output = torch.zeros(L,1)
                mask = torch.ones(L,1, dtype = torch.int32)
                for j in nonneg:
                        if j in pos:
                                output[j][0] = 1
                        else :
                                mask[j] = 0
                input = torch.cat((tokens, mask), 0)
                if i==0:
                        train_x = input
                        train_y = output
                else:
                        train_x = torch.cat((train_x, input), 0)
                        train_y = torch.cat((train_y, output),0)
        
        train_x = torch.reshape(train_x, (-1,L+T))
        train_y = torch.reshape(train_y, (-1,L,1))


        train_dataset = TensorDataset(train_x, train_y)

        B_size = 16

        train_dataloader = DataLoader(train_dataset, batch_size=B_size, shuffle=True)

        text_classifier.cuda()
        text_classifier.train()

        for epoch in range(20): 
                running_loss = 0.0
                batch_loss = 0.0
                for i, train_data in enumerate(train_dataloader):

                        optimizer.zero_grad()

                        
                        inputs, outputs = train_data
                        predicated = 0
                        for j, input in enumerate(inputs,0):
                                p = text_classifier(input.cuda()).cpu()
                                if ( j == 0 ):
                                        predicted = p
                                else:
                                        predicted = torch.cat((predicted, p), 0)
                        
                        predicted = torch.reshape(predicted, (-1, L, 1))
                        loss = loss_fun(predicted, outputs)
                        loss.backward()
                        optimizer.step()
                                
                        batch_loss += loss.item()
                        print('[%d, %5d] batch loss: %.3f' %
                                (epoch + 1, i + 1, batch_loss / 2000))
                        running_loss += batch_loss
                        batch_loss = 0.0
                print('[%d] total loss: %.3f' %
                        (epoch + 1, running_loss / 2000))


        print('Finished Training')



