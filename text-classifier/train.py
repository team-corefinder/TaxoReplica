import argparse
import os
import torch
import gensim
import dgl
import dgl.function as fn
from torch import nn
from encoder import DocuEncoder, ClassEncoder
from layer import GCN
from classifier import TextClassifier
from preprocessor import TaxoDataManager, DocumentManager
from gensim.test.utils import datapath



if __name__ == '__main__':
    print(torch.cuda.get_device_name(0))
    print(torch.version)
    print(torch.cuda.is_available())
    d_encoder = DocuEncoder()

    document = """
            An atom is the smallest unit of ordinary matter that forms a chemical element. 
            Every solid, liquid, gas, and plasma is composed of neutral or ionized atoms. 
            Atoms are extremely small, typically around 100 picometers across. 
            They are so small that accurately predicting their behavior using classical 
            physics—as if they were tennis balls, for example—is not possible due to quantum effects.
            """
    tokens = d_encoder.Tokenize(document)


    #document_representation = d_encoder.Encode(tokens)


    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(datapath('word2vec_pre_kv_c'))

    gcn_model = GCN(10, 10, 10, 16, nn.ReLU())

    class_encoder = ClassEncoder(gcn_model, word2vec_model)

    text_classifier = TextClassifier(class_encoder, (10, 768), 12)

    #mock data

    g = dgl.DGLGraph()

    document = """
            When our son was about 4 months old, our doctor said we could give him crafted cereal. 
            We bought this product and put it in his bottle.
            He loved this stuff! This cereal digests well and didn’t lock up his bowels at all. 
            We highly recommend this cereal.
            """

    #add 12 nodes


    root = '/root/Encoder/TaxoReplica/text-classifier/processed_data/'
    train_file = 'train-with-core-class-1000.jsonl'
    taxonomy_file = 'taxonomy.json'
    data_name = 'amazon'

    data_manager = TaxoDataManager(root, taxonomy_file, data_name)

    data_manager.load_all()
    dm = DocumentManager(train_file, root, 'amazon_train', d_encoder.Tokenize, d_encoder.Encode,  data_manager)
    dm.load_tokens()
    dm.load_dicts()