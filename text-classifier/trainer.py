import argparse
import os
import torch
import gensim
import dgl
import dgl.function as fn
import torch.optim as optim
import math
import time
from torch.utils.data import TensorDataset 
from torch.utils.data import DataLoader
from torch import nn
from encoder import DocuEncoder, ClassEncoder, DocumentTokenizer
from layer import GCN
from classifier import TextClassifier
from preprocessor import TaxoDataManager, DocumentManager
from gensim.test.utils import datapath
from gensim.models import word2vec




class Trainer():
        def __init__(self, dir, train_file, taxonomy_file, data_name,
                        bert_lr, others_lr, token_length, cls_length, batch_size, epoch, activation = nn.Sigmoid(), rescaling = False, test_file = None):

                self.dir = dir
                self.train_file = train_file
                if test_file != None:
                        self.test_file = test_file
                else:
                        self.test_file = 0
                self.taxonomy_file = taxonomy_file
                self.data_name = data_name
                #Document encoding input max length = token length, output length = cls_length, which is cls token's length.
                self.T = token_length
                self.C = cls_length

                #hyper parameter for training
                self.bert_lr = bert_lr
                self.others_lr = others_lr
                self.B = batch_size
                self.epoch = epoch
                self.activation = activation
                self.rescaling = rescaling
                

        def prepare_train(self):

                #document encoder
                self.d_encoder = DocuEncoder(dir)
                self.d_tokenizer = DocumentTokenizer(dir, self.T)

                #word embedding model for class encoder
                word2vec_model = word2vec.Word2Vec.load(self.dir + 'pretrained/' + 'embedding')
                self.W = word2vec_model.wv.vector_size

                #create gcn for class encoder. input_dim = W, hidden_dim = W, output_dim = W
                gcn_model = GCN(self.W, self.W, self.W, 2, nn.ReLU())

                self.class_encoder = ClassEncoder(gcn_model, word2vec_model)

                #TaxoDataManager load and manage taxonomy information of dataset.
                self.tm = TaxoDataManager( self.dir + 'training_data/'+self.data_name + '/', self.taxonomy_file, self.data_name, word2vec_model)
                self.tm.load_all()

                self.train_dm = DocumentManager(self.train_file, self.dir + 'training_data/' + self.data_name + '/' ,
                                         self.data_name + '_train', self.d_tokenizer.Tokenize,  self.tm)

                if self.test_file:
                        self.test_dm = DocumentManager(self.test_file, self.dir + 'training_data/' + self.data_name + '/' ,
                                         self.data_name + '_test', self.d_tokenizer.Tokenize,  self.tm)
                else:
                        self.test_dm = 0
                
                self.train_dm.load_tokens()
                self.train_dm.load_dicts()

                #g is graph of the taxonomy structure.
                self.g = self.tm.get_graph().to('cuda:0')
                #L is the number of the classes.
                self.L = len(self.g.nodes())

                #feature is L x W matrix, word embedding of the classes.
                self.features = self.tm.get_feature().cuda()

                self.text_classifier = TextClassifier(self.class_encoder, self.d_encoder, 
                                                        (self.W, self.C), self.T, self.g,
                                                        self.features, self.activation, self.rescaling)

                sum = 0
                for c in gcn_model.parameters():
                        sum = sum + 1
                print("GCN total Layer: ", sum)



                sum = 0
                for c in self.text_classifier.parameters():
                        sum = sum + 1
                print("Text-classfier Layer: ", sum)


                #set loss function and optimizer for training
                self.loss_fun = torch.nn.BCELoss(reduction='sum')
                self.loss_kl = torch.nn.KLDivLoss(reduction='batchmean')
                self.optimizer_kl = optim.AdamW([
                        {'params': self.text_classifier.document_encoder.parameters(), 'lr': self.bert_lr},
                        {'params': self.text_classifier.class_encoder.parameters()},
                        {'params': self.text_classifier.weight}], lr=self.others_lr)
                self.optimizer = optim.AdamW([
                        {'params': self.text_classifier.document_encoder.parameters(), 'lr': self.bert_lr},
                        {'params': self.text_classifier.class_encoder.parameters()},
                        {'params': self.text_classifier.weight}], lr=self.others_lr)
                
                train_ids = self.train_dm.get_ids()

                #generate train set
                for i, document_id in enumerate(train_ids, 0):
                        tokens = torch.tensor( self.train_dm.get_tokens(document_id) ,dtype = torch.int32)
                        tokens = torch.reshape(tokens, (-1, 1))
                        pos, nonneg = self.train_dm.get_output_label(document_id)
                        output = torch.zeros(self.L,1)
                        mask = torch.ones(self.L,1, dtype = torch.int32)

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
                
                train_x = torch.reshape(train_x, ( -1, self.L + self.T ))
                train_y = torch.reshape(train_y, ( -1, self.L, 1 ))


                train_dataset = TensorDataset(train_x, train_y)


                self.train_dataloader = DataLoader(train_dataset, batch_size=self.B, shuffle=True)
                
        
        def train(self):
                print("Start training! bert learning rate: %f, other learning rate: %f, epoch: %d, batch size: %d"
                        %(self.bert_lr, self.others_lr, self.epoch, self.B))
                self.text_classifier.cuda()
                self.text_classifier.train()

                for epoch in range(self.epoch): 
                        start = time.time()
                        running_loss = 0.0
                        batch_loss = 0.0
                        self.optimizer.zero_grad()
                        self.optimizer_kl.zero_grad()
                        for i, train_data in enumerate(self.train_dataloader):



                                inputs, outputs = train_data
                                predicted = self.text_classifier(inputs.cuda())
                                loss = self.loss_fun(predicted, outputs.cuda())
                                batch_loss += loss.item()
                                if (i+1) % 25 == 0:
                                        print('Start self-training...')
                                        weight = predicted**2 / predicted.sum(axis=0)
                                        #q = (weight.T / weight.sum(axis=1).T)
                                        weight_1 = (1-predicted)**2 / (1-predicted).sum(axis=0)
                                        q = weight / (weight + weight_1)
                                        # shape of predicted = (Batch size, the number of labels, 1)
                                        loss_kl = self.loss_kl(q, predicted)
                                        (loss+loss_kl).backward()
                                        self.optimizer_kl.step()
                                        self.optimizer_kl.zero_grad()
                                        print('[%d, %5d] KL loss: %.3f' %
                                              (epoch + 1, i + 1, loss_kl.item()))
                                        
                                else:
                                        loss.backward()


                                if (i+1)%8 == 0 :
                                        #print("optimized!")
                                        self.optimizer.step()
                                        self.optimizer.zero_grad()
                                        print('[%d, %5d] batch loss: %.3f' %
                                                (epoch + 1, i + 1, batch_loss ))
                                running_loss += batch_loss
                                batch_loss = 0.0

                                


                        print('[%d] total loss: %.3f' %
                                (epoch + 1, running_loss ))
                        print('elapsed time : %f'%(time.time()-start))

                print('Finished Training')




        

if __name__ == '__main__':
        
        #dir = './'

        root = os.path.dirname(os.path.abspath(__file__)) 

        if not os.path.isdir(root + "/data"):
                url = "https://drive.google.com/drive/folders/1K6oXC2lKZdNcFaPVuCnozHojhBLSfcfb"
                output ="data"
                gdown.download_folder(url, output= output)

        dir = root + "/data/"

        """
        #DBPedia dataset
        train_file = 'DBPEDIA-coreclass-45000.jsonl'
        taxonomy_file = 'taxonomy.json'
        data_name = 'DBPEDIA'
        """

        
        #amazon dataset
        train_file = 'train-with-core-class-1000.jsonl'
        taxonomy_file = 'taxonomy.json'
        data_name = 'amazon'
        
        bert_lr = 5e-5
        others_lr = 4e-3
        token_length = 500
        batch_size =4
        epoch = 20
        cls_length = 768
        
        #default activation function is Sigmoid
        activation = nn.Sigmoid()
        #activation = nn.Softmax(dim = 1)

        rescaling = False

        trainer = Trainer(dir, train_file, taxonomy_file, data_name,
                        bert_lr, others_lr, token_length, cls_length, 
                        batch_size, epoch, activation, rescaling)

        trainer.prepare_train()
        trainer.train()
