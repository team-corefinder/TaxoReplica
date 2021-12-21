import argparse
import os
from networkx.readwrite.json_graph import jit
import torch
import gensim
import dgl
import dgl.function as fn
import torch.optim as optim
import math
import time
import copy
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import nn
from encoder import DocuEncoder, ClassEncoder, DocumentTokenizer
from layer import GCN
from classifier import TextClassifier
from preprocessor import TaxoDataManager, DocumentManager
from gensim.test.utils import datapath
from gensim.models import word2vec
from tqdm import tqdm, trange

def safe_div(a, b, epsilon=1e-8):
    return a / b.clamp(min=epsilon)

def safe_log(a, epsilon=1e-8):
    return torch.log(a.clamp(min=epsilon))

class No_GCN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(No_GCN, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(in_dim, out_dim))

    def forward(self, g, features):
        for layer in self.layers:
            features = layer(features)
        return features

class Trainer:
    def __init__(
        self,
        dir,
        train_file,
        taxonomy_file,
        data_name,
        bert_lr,
        others_lr,
        token_length,
        cls_length,
        batch_size,
        epoch,
        activation=nn.Sigmoid(),
        rescaling=False,
        test_file=None,
    ):

        self.dir = dir
        self.train_file = train_file
        if test_file != None:
            self.test_file = test_file
        else:
            self.test_file = 0
        self.taxonomy_file = taxonomy_file
        self.data_name = data_name
        # Document encoding input max length = token length, output length = cls_length, which is cls token's length.
        self.T = token_length
        self.C = cls_length

        # hyper parameter for training
        self.bert_lr = bert_lr
        self.others_lr = others_lr
        self.B = batch_size
        self.epoch = epoch
        self.activation = activation
        self.rescaling = rescaling

    def F1_evaluation(self, true_labels, prediction, threshold):
        N = true_labels.shape[0]
        true_labels = torch.reshape(true_labels, (N, -1))
        prediction = torch.reshape(prediction, (N, -1))
        sum = 0.0
        for i in range(N):
            prediction[i][0] = 0
            mask = prediction[i] >= threshold
            pred_label = torch.flatten(torch.nonzero(mask))
            numerator = len(set(true_labels[i].tolist()) & set(pred_label.tolist()))
            denominator = true_labels.shape[1] + pred_label.shape[0]
            sum += numerator * 2 / denominator

        return sum

    def PN_evaluation(self, true_labels, prediction, num):
        N = true_labels.shape[0]
        true_labels = torch.reshape(true_labels, (N, -1))
        prediction = torch.reshape(prediction, (N, -1))
        L = prediction.shape[1]
        ids = list(range(L))
        sum = 0.0
        prediction = prediction.cpu()
        for i in range(N):
            prediction[i][0] = 0
            top_n = sorted(zip(prediction[i].tolist(), ids), reverse=True)[:num]
            top_n = [j for i, j in top_n]
            numerator = len(set(true_labels[i].tolist()) & set(top_n))
            denominator = min(true_labels.shape[1], num)
            sum += numerator / denominator
        return sum
    
    def MRR_evaluation(self, true_labels, prediction):
        N = true_labels.shape[0]
        true_labels = torch.reshape(true_labels, (N, -1))
        prediction = torch.reshape(prediction, (N, -1))
        L = prediction.shape[1]
        ids = list(range(L))
        acc = 0.0
        prediction = prediction.cpu()
        for i in range(N):
            prediction[i][0] = 0
            top_n = sorted(zip(prediction[i].tolist(), ids), reverse=True)
            top_n = [b for a, b in top_n]
            ranks = [1/rank for rank, label in enumerate(top_n, 1) if label in true_labels[i].tolist()]
            numerator = sum(ranks)
            denominator = true_labels.shape[1]
            acc += numerator / denominator
        return acc

    def prepare_train(self):

        # document encoder
        self.d_encoder = DocuEncoder(dir)
        self.d_tokenizer = DocumentTokenizer(dir, self.T)

        # word embedding model for class encoder
        word2vec_model = word2vec.Word2Vec.load(self.dir + "pretrained/" + "embedding")
        self.W = word2vec_model.wv.vector_size

        # create gcn for class encoder. input_dim = W, hidden_dim = W, output_dim = W
        # gcn_model = GCN(self.W, self.W, self.W, 2, nn.ReLU())
        gcn_model = No_GCN(self.W, self.W)

        self.class_encoder = ClassEncoder(gcn_model, word2vec_model)

        # TaxoDataManager load and manage taxonomy information of dataset.
        self.tm = TaxoDataManager(
            self.dir + "training_data/" + self.data_name + "/",
            self.taxonomy_file,
            self.data_name,
            word2vec_model,
        )
        self.tm.load_all(normalize = True)

        self.train_dm = DocumentManager(
            self.train_file,
            self.dir + "training_data/" + self.data_name + "/",
            self.data_name + "_train",
            self.d_tokenizer.Tokenize,
            self.tm,
        )

        if self.test_file:
            self.test_dm = DocumentManager(
                self.test_file,
                self.dir + "training_data/" + self.data_name + "/",
                self.data_name + "_test",
                self.d_tokenizer.Tokenize,
                self.tm,
            )
        else:
            self.test_dm = 0

        self.train_dm.load_tokens()
        self.train_dm.load_dicts()

        # g is graph of the taxonomy structure.
        self.g = self.tm.get_graph().to("cuda:0")
        # L is the number of the classes.
        self.L = len(self.g.nodes())

        # feature is L x W matrix, word embedding of the classes.
        self.features = self.tm.get_feature().cuda()

        self.text_classifier = TextClassifier(
            self.class_encoder,
            self.d_encoder,
            (self.W, self.C),
            self.T,
            self.g,
            self.features,
            self.activation,
            self.rescaling,
        )
        
        # sum = 0
        # for c in gcn_model.parameters():
        #     sum = sum + 1
        # print("GCN total Layer: ", sum)

        sum = 0
        for c in self.text_classifier.parameters():
            sum = sum + 1
        print("Text-classfier Layer: ", sum)

        # set loss function and optimizer for training
        self.loss_fun = torch.nn.BCELoss(reduction="sum")
        self.loss_kl = torch.nn.KLDivLoss(reduction="batchmean")
        self.optimizer_kl = optim.AdamW(
            [
                {
                    "params": self.text_classifier.document_encoder.parameters(),
                    "lr": 1e-6,
                },
                {"params": self.text_classifier.class_encoder.parameters()},
                {"params": self.text_classifier.weight},
            ],
            lr=5e-4,
        )
        self.optimizer = optim.AdamW(
            [
                {
                    "params": self.text_classifier.document_encoder.parameters(),
                    "lr": self.bert_lr,
                },
                {"params": self.text_classifier.class_encoder.parameters()},
                {"params": self.text_classifier.weight},
            ],
            lr=self.others_lr,
        )

        train_ids = list(self.train_dm.get_ids())

        # generate train set
        for i, document_id in enumerate(train_ids, 0):
            tokens = torch.tensor(
                self.train_dm.get_tokens(document_id), dtype=torch.int32
            )
            tokens = torch.reshape(tokens, (-1, 1))
            pos, nonneg = self.train_dm.get_output_label(document_id)
            output = torch.zeros(self.L + 3, 1)
            mask = torch.ones(self.L, 1, dtype=torch.int32)
            categories = self.train_dm.get_categories(document_id)
            for num, category in enumerate(categories, 0):
                output[self.L + num] = category

            for j in nonneg:
                if j in pos:
                    output[j][0] = 1
                else:
                    mask[j] = 0

            #ignore root node
            mask[0] = 0

            input = torch.cat((tokens, mask), 0)
            if i == 0:
                train_x = input
                train_y = output
            else:
                train_x = torch.cat((train_x, input), 0)
                train_y = torch.cat((train_y, output), 0)

        train_x = torch.reshape(train_x, (-1, self.L + self.T))
        train_y = torch.reshape(train_y, (-1, self.L + 3, 1))

        self.data_size = train_x.shape[0]
        train_dataset = TensorDataset(train_x, train_y)

        self.train_dataloader = DataLoader(
            train_dataset, batch_size=self.B, shuffle=True
        )

    def save_model(self, path=None):
        if path is None:
            path = f'{self.dir}trained/text-classifier-{self.data_name}.pt'
        torch.save(self.text_classifier.state_dict(), path)
        print("model saved!")
        return

    def load_pretrained_model(self, path=None):
        if path is None:
            path = f'{self.dir}trained/text-classifier-{self.data_name}.pt'
        self.text_classifier.load_state_dict(torch.load(path))
        self.text_classifier.eval()
        return
      
    def target_distribution(self, prediction):
      prediction = prediction.reshape(4, self.L)
      weight = safe_div(prediction ** 2, prediction.sum(axis=0))
      weight_1 = safe_div((1 - prediction) **2, (1 - prediction).sum(axis=0))
      result = safe_div(weight, (weight + weight_1))
      # return result
      return result.reshape(4, self.L, 1)
    
    def kl_div_loss(self, predicted, target):
      # predicted = predicted.reshape(4, self.L)
      # target = target.reshape(4, self.L)
      return (target * safe_log(safe_div(target, predicted))).sum()

    def self_train(self, patience, threshold):
        print(
            "Start training! bert learning rate: %f, other learning rate: %f, epoch: %d, batch size: %d"
            % (self.bert_lr, self.others_lr, self.epoch, self.B)
        )
        self.text_classifier.cuda()
        self.text_classifier.train()
        
        max_accuracy = 0.0
        patience_count = 0

        for epoch in range(self.epoch):
            start = time.time()
            running_loss = 0.0
            
            running_accuracy = 0.0
            p1_running_accuracy = 0.0
            p3_running_accuracy = 0.0
            
            self.optimizer_kl.zero_grad()
            
            num_batch = len(self.train_dataloader)
            # update_interval = int(num_batch / 2)
            update_interval = 450
            
            for i, train_data in enumerate(tqdm(self.train_dataloader)):
                if i % update_interval == 0:
                    for j, data in enumerate(tqdm(self.train_dataloader, leave=None, total=update_interval), start=i):
                        if j == i + update_interval:
                          break
                        inputs, outputs = data
                        with torch.no_grad():
                            p = self.text_classifier(inputs.cuda())
                        outputs[:, :-3] = self.target_distribution(p)
                        
                inputs, outputs = train_data
                predicted = self.text_classifier(inputs.cuda())
                offset = 4 * i
                target = outputs[:, :-3].cuda()
                # target = self.train_y[offset:offset+4, :-3].cuda()
                
                loss_kl = self.kl_div_loss(predicted, target)
                loss_kl.backward()
                running_loss += loss_kl.item()

                true_labels = outputs[:, self.L :, :]
                running_accuracy += self.F1_evaluation(true_labels, predicted, threshold)
                p1_running_accuracy += self.PN_evaluation(true_labels, predicted, 1)
                p3_running_accuracy += self.PN_evaluation(true_labels, predicted, 3)
                
                if i % update_interval == update_interval-1 or i == num_batch-1:
                  self.optimizer_kl.step()
                  self.optimizer_kl.zero_grad()
                
            print(
                "[%d] total datat: %d, total loss: %.3f, f1 accuracy: %.3f%% ,p1 accuracy: %.3f%%, p3 accuracy: %.3f%%"
                % (
                    epoch + 1,
                    self.data_size,
                    running_loss,
                    running_accuracy * 100 / self.data_size,
                    p1_running_accuracy * 100 / self.data_size,
                    p3_running_accuracy * 100 / self.data_size,
                )
            )
            print("[%d] elapsed time : %f" % (epoch + 1, time.time() - start))

            if running_accuracy > max_accuracy:
                self.save_model(
                  f'{self.dir}trained/text-classifier-{self.data_name}-NoGNN-with-st.pt')
                max_accuracy = running_accuracy
                patience_count = 0
            else:
                patience_count += 1

            if patience_count >= patience:
                print("Finished Training")
                return

        print("Finished Training")

    def train(self, patience, threshold):
        print(
            "Start training! bert learning rate: %f, other learning rate: %f, epoch: %d, batch size: %d"
            % (self.bert_lr, self.others_lr, self.epoch, self.B)
        )
        self.text_classifier.cuda()
        self.text_classifier.train()
        max_accuracy = 0.0
        patience_count = 0

        for epoch in range(self.epoch):
            start = time.time()
            running_loss = 0.0
            batch_loss = 0.0

            batch_accuracy = 0.0
            running_accuracy = 0.0

            p1_batch_accuracy = 0.0
            p1_running_accuracy = 0.0

            p3_batch_accuracy = 0.0
            p3_running_accuracy = 0.0

            self.optimizer.zero_grad()
            for i, train_data in enumerate(self.train_dataloader):

                inputs, outputs = train_data
                true_labels = outputs[:, self.L :, :]
                outputs = outputs[:, : self.L, :]

                predicted = self.text_classifier(inputs.cuda())
                loss = self.loss_fun(predicted, outputs.cuda())
                loss.backward()

                accuracy = self.F1_evaluation(true_labels, predicted, threshold)
                batch_accuracy += accuracy

                accuracy = self.PN_evaluation(true_labels, predicted, 1)
                p1_batch_accuracy += accuracy

                accuracy = self.PN_evaluation(true_labels, predicted, 3)
                p3_batch_accuracy += accuracy

                batch_loss += loss.item()

                if (i + 1) % 8 == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    print(
                        "[%d, %5d] batch loss: %.3f f1 accuracy : %.3f%% p1 accuracy: %.3f%% p3 accuracy: %.3f%%"
                        % (
                            epoch + 1,
                            i + 1,
                            batch_loss,
                            batch_accuracy * 100 / (self.B * 7 + predicted.shape[0]),
                            p1_batch_accuracy * 100 / (self.B * 7 + predicted.shape[0]),
                            p3_batch_accuracy * 100 / (self.B * 7 + predicted.shape[0]),
                        )
                    )
                    running_accuracy += batch_accuracy
                    p1_running_accuracy += p1_batch_accuracy
                    p3_running_accuracy += p3_batch_accuracy
                    batch_accuracy = 0.0
                    p1_batch_accuracy = 0.0
                    p3_batch_accuracy = 0.0
                    running_loss += batch_loss
                    batch_loss = 0.0
            print(
                "[%d] total datat: %d, total loss: %.3f, f1 accuracy: %.3f%% ,p1 accuracy: %.3f%%, p3 accuracy: %.3f%%"
                % (
                    epoch + 1,
                    self.data_size,
                    running_loss,
                    running_accuracy * 100 / self.data_size,
                    p1_running_accuracy * 100 / self.data_size,
                    p3_running_accuracy * 100 / self.data_size,
                )
            )
            print("[%d] elapsed time : %f" % (epoch + 1, time.time() - start))

            if running_accuracy > max_accuracy:
                self.save_model()
                max_accuracy = running_accuracy
                patience_count = 0
            else:
                patience_count += 1

            if patience_count >= patience:
                print("Finished Training")
                return

        print("Finished Training")

    def evaluation(self, threshold):
            print(
                "Start evaluation!  batch size: %d"
                % (self.B)
            )
            self.text_classifier.cuda()
            self.text_classifier.eval()

            start = time.time()
            running_loss = 0.0
            batch_loss = 0.0

            batch_accuracy = 0.0
            running_accuracy = 0.0

            p1_batch_accuracy = 0.0
            p1_running_accuracy = 0.0

            p3_batch_accuracy = 0.0
            p3_running_accuracy = 0.0

            for i, train_data in enumerate(self.train_dataloader):

                inputs, outputs = train_data
                true_labels = outputs[:, self.L :, :]
                outputs = outputs[:, : self.L, :]

                predicted = self.text_classifier(inputs.cuda())

    

                accuracy = self.F1_evaluation(true_labels, predicted, threshold)
                batch_accuracy += accuracy

                accuracy = self.PN_evaluation(true_labels, predicted, 1)
                p1_batch_accuracy += accuracy

                accuracy = self.PN_evaluation(true_labels, predicted, 3)
                p3_batch_accuracy += accuracy

                if (i + 1) % 8 == 0:
                    print(
                        "[%5d]f1 accuracy : %.3f%% p1 accuracy: %.3f%% p3 accuracy: %.3f%%, num_data: %d"
                        % (
                            i + 1,
                            batch_accuracy * 100 / (self.B * 7 + predicted.shape[0]),
                            p1_batch_accuracy * 100 / (self.B * 7 + predicted.shape[0]),
                            p3_batch_accuracy * 100 / (self.B * 7 + predicted.shape[0]),
                            self.B * 7 + predicted.shape[0]
                        )
                    )
                    running_accuracy += batch_accuracy
                    p1_running_accuracy += p1_batch_accuracy
                    p3_running_accuracy += p3_batch_accuracy
                    batch_accuracy = 0.0
                    p1_batch_accuracy = 0.0
                    p3_batch_accuracy = 0.0

            print(
                "[%s] f1 accuracy: %.3f%% ,p1 accuracy: %.3f%%, p3 accuracy: %.3f%%"
                % (
                    self.data_name,
                    running_accuracy * 100 / self.data_size,
                    p1_running_accuracy * 100 / self.data_size,
                    p3_running_accuracy * 100 / self.data_size,
                )
            )
            print("elapsed time : %f" % ( time.time() - start))
            print("Finished Evaluation")


if __name__ == "__main__":

    # dir = './'

    root = os.path.dirname(os.path.abspath(__file__))

    if not os.path.isdir(root + "/data"):
        url = "https://drive.google.com/drive/folders/1K6oXC2lKZdNcFaPVuCnozHojhBLSfcfb"
        output = "data"
        gdown.download_folder(url, output=output)

    dir = root + "/data/"

    """
        #DBPedia dataset
        train_file = 'DBPEDIA-coreclass-45000.jsonl'
        taxonomy_file = 'taxonomy.json'
        data_name = 'DBPEDIA'
        """

    # amazon dataset
    train_file = "amazon-coreclass-45000.jsonl"
    taxonomy_file = "taxonomy.json"
    data_name = "amazon"

    bert_lr = 5e-5
    others_lr = 4e-3
    token_length = 500
    batch_size = 4
    epoch = 20
    cls_length = 768

    # default activation function is Sigmoid
    activation = nn.Sigmoid()
    # activation = nn.Softmax(dim = 1)

    rescaling = False
    is_train = False
    is_self_train = True

    trainer = Trainer(
        dir,
        train_file,
        taxonomy_file,
        data_name,
        bert_lr,
        others_lr,
        token_length,
        cls_length,
        batch_size,
        epoch,
        activation,
        rescaling,
    )

    trainer.prepare_train()
    if is_train : 
        trainer.train(patience=3, threshold=0.95)
    elif is_self_train :
        trainer.load_pretrained_model(
          f'{trainer.dir}trained/text-classifier-{trainer.data_name}-NoGNN.pt')
        trainer.self_train(patience=5, threshold=0.95)
    else : 
        trainer.load_pretrained_model()
        trainer.evaluation(threshold = 0.95)

