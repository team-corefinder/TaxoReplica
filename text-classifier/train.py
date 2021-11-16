import argparse
import os
import torch
import gensim
import dgl
import dgl.function as fn
import torch.optim as optim
from torch import nn
from encoder import DocuEncoder, ClassEncoder
from layer import GCN
from classifier import TextClassifier
from preprocessor import TaxoDataManager, DocumentManager
from gensim.test.utils import datapath
from gensim.models import word2vec



if __name__ == '__main__':
        print(torch.cuda.is_available())


        root = '/root/text-classifier/TaxoReplica/text-classifier/'
        train_file = 'train-with-core-class-1000.jsonl'
        taxonomy_file = 'taxonomy.json'
        data_name = 'amazon'

        d_encoder = DocuEncoder()
        #document_representation = d_encoder.Encode(tokens)

        word2vec_model = word2vec.Word2Vec.load(root + 'pretrained/' + 'embedding')
        W = word2vec_model.wv.vector_size

        gcn_model = GCN(W, W, W, 2, nn.ReLU())

        class_encoder = ClassEncoder(gcn_model, word2vec_model)

        text_classifier = TextClassifier(class_encoder, (W, 768))

        g = dgl.DGLGraph()


        data_manager = TaxoDataManager(root + 'processed_data/', taxonomy_file, data_name, word2vec_model)
        data_manager.load_all()
        dm = DocumentManager(train_file, root + 'processed_data/', 'amazon_train', d_encoder.Tokenize, d_encoder.Encode,  data_manager)
        dm.load_tokens()
        dm.load_dicts()

        #for c in text_classifier.parameters():
        #        print(c)

        loss_fun = torch.nn.BCELoss(reduction='sum')
        optimizer = optim.AdamW(text_classifier.parameters(), lr=0.004)
        g = data_manager.get_graph()
        L = len(g.nodes())
        features = data_manager.get_feature()

        train_data = dm.get_ids()


        for epoch in range(2): 
                running_loss = 0.0

                for i, document_id in enumerate(train_data, 0):

                        optimizer.zero_grad()

                        tokens= dm.get_tokens(document_id)
                        target = dm.get_output_label(document_id)

                        # p = (L x 1)
                        d = torch.reshape(torch.tensor(tokens), (-1, 1))
                        p = text_classifier(g,d, features)

                        pos, nonneg = target
                        outputs = torch.zeros(p.shape)

                        for j in nonneg:
                                outputs[j][0] = 1
                        #        if not (i in pos):
                        #                p[i][0] = 1

                        loss = loss_fun(p, outputs)
                        loss.backward()
                        optimizer.step()

                        # print statistics
                        running_loss += loss.item()
                        if i % 64 == 63:
                                print('[%d, %5d] loss: %.3f' %
                                        (epoch + 1, i + 1, running_loss / 2000))
                                running_loss = 0.0


        print('Finished Training')



