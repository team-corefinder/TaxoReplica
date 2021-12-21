import argparse
import os
from networkx.readwrite.json_graph import jit
from numpy.core.numeric import False_
import torch
import gensim
import dgl
import dgl.function as fn
import torch.optim as optim
import math
import time
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import nn
from encoder import DocuEncoder, ClassEncoder, DocumentTokenizer
from layer import GCN
from classifier import TextClassifier
from preprocessor3 import TaxoDataManager, DocumentManager
from gensim.test.utils import datapath
from gensim.models import word2vec
from sklearn.metrics import f1_score


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
            self.test_file = None
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

    def F1_evaluation(self, true_labels, prediction, threshold, use_depth=False):
        N = true_labels.shape[0]
        true_labels = torch.reshape(true_labels, (N, -1))
        prediction = torch.reshape(prediction, (N, -1))
        sum = 0.0
        L = prediction.shape[1]
        ids = list(range(L))
        macro_sum = 0.0
        micro_sum = 0.0

        for i in range(N):
            prediction[i][0] = 0
            #choose three label according to depth
            if use_depth:
                pred_label = [0, 0, 0]
                top_n = sorted(zip(prediction[i].tolist(), ids), reverse=True)
                for dep in range(1,4):
                    dep_list = self.tm.get_depth_list(dep)
                    dep_top = [label for p, label in top_n if label in dep_list]
                    pred_label[dep-1] = dep_top[0]
                numerator = len(set(true_labels[i].tolist()) & set(pred_label))
                denominator = true_labels.shape[1] + len(pred_label)
                f1_macro = f1_score(true_labels[i].tolist(), pred_label, average='macro')
                f1_micro = f1_score(true_labels[i].tolist(), pred_label, average='micro')
            else :
                mask = prediction[i] >= threshold
                pred_label = torch.flatten(torch.nonzero(mask))
                numerator = len(set(true_labels[i].tolist()) & set(pred_label.tolist()))
                denominator = true_labels.shape[1] + pred_label.shape[0]
                #f1_macro = f1_score(true_labels[i].tolist(), pred_label.tolist()[:3], average='macro')
                #f1_micro = f1_score(true_labels[i].tolist(), pred_label.tolist()[:3], average='micro')

            sum += numerator * 2 / denominator
            #macro_sum += f1_macro
            #micro_sum += f1_micro

        return [sum, macro_sum, micro_sum]

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
        self.W = 300
        # create gcn for class encoder. input_dim = W, hidden_dim = W, output_dim = W
        #gcn_model = GCN(self.W, self.W, self.W, 2, nn.ReLU())
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

        if self.test_file != None:
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
        self.test_dm.load_tokens()
        self.test_dm.load_dicts()

        # g is graph of the taxonomy structure.
        self.g = self.tm.get_graph().to("cuda:0")
        # L is the number of the classes.
        self.L = len(self.g.nodes())

        # feature is L x W matrix, word embedding of the classes.
        self.features = self.tm.get_feature().cuda()
        print(self.features.shape)

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

        sum = 0
        for c in gcn_model.parameters():
            sum = sum + 1
        print("GCN total Layer: ", sum)

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
                    "lr": self.bert_lr,
                },
                {"params": self.text_classifier.class_encoder.parameters()},
                {"params": self.text_classifier.weight},
            ],
            lr=self.others_lr,
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

        train_ids = self.train_dm.get_ids()

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
        print("train file is loaded!")
        
        #self.test_size = 15000

        #self.data_size -= int(self.data_size/3)

        #test_x = train_x[:self.test_size, :]
        #test_y = train_y[:self.test_size, :, :]

        #train_x = train_x[self.test_size:, :]
        #train_y = train_y[self.test_size:, :, :]


        test_ids = self.test_dm.get_ids()

        # generate test set
        for i, document_id in enumerate(test_ids, 0):
            tokens = torch.tensor(
                self.test_dm.get_tokens(document_id), dtype=torch.int32
            )
            tokens = torch.reshape(tokens, (-1, 1))
            pos, nonneg = self.test_dm.get_output_label(document_id)
            output = torch.zeros(self.L + 3, 1)
            mask = torch.ones(self.L, 1, dtype=torch.int32)
            categories = self.test_dm.get_categories(document_id)
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
                test_x = input
                test_y = output
            else:
                test_x = torch.cat((test_x, input), 0)
                test_y = torch.cat((test_y, output), 0)

        test_x = torch.reshape(test_x, (-1, self.L + self.T))
        test_y = torch.reshape(test_y, (-1, self.L + 3, 1))

        self.test_size = test_x.shape[0]
        print("test file is loaded!")

        train_dataset = TensorDataset(train_x, train_y)
        test_dataset = TensorDataset(test_x, test_y)

        self.train_dataloader = DataLoader(
            train_dataset, batch_size=self.B, shuffle=True
        )
        self.test_dataloader = DataLoader(
            test_dataset, batch_size=self.B, shuffle=False
        )

    def save_model(self):
        torch.save(
            self.text_classifier.state_dict(), self.dir + "trained/text-classifier-"+self.data_name + "-without-GNN3.pt"
        )
        print("model saved!")
        return

    def load_pretrained_model(self):
        self.text_classifier.load_state_dict(
            torch.load(self.dir + "trained/text-classifier-"+self.data_name + "-without-GNN3.pt")

        )
        #            torch.load(self.dir + "trained/text-classifier-with-testset2.pt")
        # self.dir + "trained/text-classifier-"+self.data_name + "-without-GNN.pt"
        self.text_classifier.eval()
        return

    def self_train(self):
        print(
            "Start training! bert learning rate: %f, other learning rate: %f, epoch: %d, batch size: %d"
            % (self.bert_lr, self.others_lr, self.epoch, self.B)
        )
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
                true_labels = outputs[:, self.L :, :]
                outputs = outputs[:, : self.L, :]
                predicted = self.text_classifier(inputs.cuda())
                loss = self.loss_fun(predicted, outputs.cuda())
                batch_loss += loss.item()
                if (i + 1) % 25 == 0:
                    print("Start self-training...")
                    weight = predicted ** 2 / predicted.sum(axis=0)
                    # q = (weight.T / weight.sum(axis=1).T)
                    weight_1 = (1 - predicted) ** 2 / (1 - predicted).sum(axis=0)
                    q = weight / (weight + weight_1)
                    # shape of predicted = (Batch size, the number of labels, 1)
                    loss_kl = self.loss_kl(q, predicted)
                    (loss + loss_kl).backward()
                    self.optimizer_kl.step()
                    self.optimizer_kl.zero_grad()
                    print(
                        "[%d, %5d] KL loss: %.3f" % (epoch + 1, i + 1, loss_kl.item())
                    )

                else:
                    loss.backward()

                if (i + 1) % 8 == 0:
                    # print("optimized!")
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    print("[%d, %5d] batch loss: %.3f" % (epoch + 1, i + 1, batch_loss))
                running_loss += batch_loss
                batch_loss = 0.0

            print("[%d] train total loss: %.3f" % (epoch + 1, running_loss))
            print("elapsed time : %f" % (time.time() - start))

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

            MRR_batch_accuracy = 0.0
            MRR_running_accuracy = 0.0

            self.optimizer.zero_grad()
            for i, train_data in enumerate(self.train_dataloader):

                inputs, outputs = train_data
                true_labels = outputs[:, self.L :, :]
                outputs = outputs[:, : self.L, :]

                predicted = self.text_classifier(inputs.cuda())
                loss = self.loss_fun(predicted, outputs.cuda())
                loss.backward()

                accuracy = self.F1_evaluation(true_labels, predicted, threshold, True)
                batch_accuracy += accuracy[0]

                accuracy = self.PN_evaluation(true_labels, predicted, 1)
                p1_batch_accuracy += accuracy

                accuracy = self.PN_evaluation(true_labels, predicted, 3)
                p3_batch_accuracy += accuracy

                accuracy = self.MRR_evaluation(true_labels, predicted)
                MRR_batch_accuracy += accuracy

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

            ##test set!
            with torch.no_grad(): 
                test_loss = 0.0
                test_acc = 0.0
                test_p1_acc = 0.0
                test_p3_acc = 0.0
                test_batch_loss =0.0
                test_batch_acc = 0.0
                test_p1_batch_acc = 0.0
                test_p3_batch_acc = 0.0
                for i, test_data in enumerate(self.test_dataloader):

                    inputs, outputs = test_data
                    true_labels = outputs[:, self.L :, :]
                    outputs = outputs[:, : self.L, :]

                    predicted = self.text_classifier(inputs.cuda())
                    #loss = self.loss_fun(predicted, outputs.cuda())

                    accuracy = self.F1_evaluation(true_labels, predicted, threshold, use_depth = False)
                    test_batch_acc += accuracy[0]

                    accuracy = self.PN_evaluation(true_labels, predicted, 1)
                    test_p1_batch_acc += accuracy

                    accuracy = self.PN_evaluation(true_labels, predicted, 3)
                    test_p3_batch_acc += accuracy

                    test_batch_loss += loss.item()

                    if (i + 1) % 8 == 0:
                        print(
                            "[%d, %5d] test batch loss: %.3f f1 accuracy : %.3f%% p1 accuracy: %.3f%% p3 accuracy: %.3f%%"
                            % (
                                epoch + 1,
                                i + 1,
                                test_batch_loss,
                                test_batch_acc * 100 / (self.B * 7 + predicted.shape[0]),
                                test_p1_batch_acc * 100 / (self.B * 7 + predicted.shape[0]),
                                test_p3_batch_acc * 100 / (self.B * 7 + predicted.shape[0]),
                            )
                        )
                        test_acc += test_batch_acc
                        test_p1_acc += test_p1_batch_acc
                        test_p3_acc += test_p3_batch_acc
                        test_batch_acc = 0.0
                        test_p1_batch_acc= 0.0
                        test_p3_batch_acc= 0.0
                        test_loss += test_batch_loss
                        test_batch_loss = 0.0
                print(
                    "[epoch : %d, train] total data: %d, total loss: %.3f, f1 accuracy: %.3f%% ,p1 accuracy: %.3f%%, p3 accuracy: %.3f%%"
                    % (
                        epoch + 1,
                        self.data_size,
                        running_loss,
                        running_accuracy * 100 / self.data_size,
                        p1_running_accuracy * 100 / self.data_size,
                        p3_running_accuracy * 100 / self.data_size,
                    )
                )

                print(
                    "[epoch : %d, test] total data: %d, total loss: %.3f, f1 accuracy: %.3f%% ,p1 accuracy: %.3f%%, p3 accuracy: %.3f%%"
                    % (
                        epoch + 1,
                        self.test_size,
                        test_loss,
                        test_acc * 100 / self.test_size,
                        test_p1_acc * 100 / self.test_size,
                        test_p3_acc * 100 / self.test_size,
                    )
                )
                print("[%d] elapsed time : %f" % (epoch + 1, time.time() - start))

            if test_acc> max_accuracy:
                self.save_model()
                max_accuracy = test_acc
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

            with open(self.dir + self.data_name + '_eval_result.txt', "w") as fout:

                start = time.time()
                running_loss = 0.0
                batch_loss = 0.0

                batch_accuracy = 0.0
                running_accuracy = 0.0

                macro_batch_accuracy = 0.0
                macro_running_accuracy = 0.0

                micro_batch_accuracy = 0.0
                micro_running_accuracy = 0.0

                p1_batch_accuracy = 0.0
                p1_running_accuracy = 0.0

                p3_batch_accuracy = 0.0
                p3_running_accuracy = 0.0

                MRR_batch_accuracy = 0.0
                MRR_running_accuracy = 0.0

                for i, train_data in enumerate(self.test_dataloader):

                    inputs, outputs = train_data
                    true_labels = outputs[:, self.L :, :]
                    outputs = outputs[:, : self.L, :]

                    predicted = self.text_classifier(inputs.cuda())

        

                    accuracy = self.F1_evaluation(true_labels, predicted, threshold, use_depth = False)
                    batch_accuracy += accuracy[0]
                    macro_batch_accuracy += accuracy[1]
                    micro_batch_accuracy += accuracy[2]

                    accuracy = self.PN_evaluation(true_labels, predicted, 1)
                    p1_batch_accuracy += accuracy

                    accuracy = self.PN_evaluation(true_labels, predicted, 3)
                    p3_batch_accuracy += accuracy

                    accuracy = self.MRR_evaluation(true_labels, predicted)
                    MRR_batch_accuracy += accuracy

                    V = predicted.shape[0]
                    predicted = predicted.reshape((V, -1))
                    true_labels = true_labels.reshape((V,-1))
                    for j in range(V):
                        pos_labels = torch.nonzero(outputs[j].reshape((-1))).reshape((-1))
                        pos_labels = list(map(self.tm.label_from_id, pos_labels.tolist()))
                        raw_list = self.d_tokenizer.DecodeToken(inputs[j])
                        raw_list = [token for token in raw_list if token != '[unused0]']
                        raw_text = ' '.join(raw_list)
                        ids = list(range(self.L))
                        top_n = sorted(zip(predicted[j].tolist(), ids), reverse=True)
                        top5 = top_n[:5]
                        top_n = [b for a, b in top_n]
                        ranks = [(self.tm.label_from_id(label), rank) for rank, label in enumerate(top_n, 1) if label in true_labels[j].tolist()]
                        top5 = list(map(lambda x: (self.tm.label_from_id(x[1]), x[0]), top5))
                        #category_labels = list(map(self.tm.label_from_id, true_labels[j].tolist()))
                        
                        fout.write(f"raw text: {raw_text} (true_label, rank): {ranks}, pos_labels:{pos_labels}, top5 probability: {top5}\n")


                    if (i + 1) % 8 == 0:
                        print(
                            "[%5d]f1 accuracy : %.3f%%, f1 macro accuracy : %.3f%%, f1 micro accuracy : %.3f%% p1 accuracy: %.3f%% p3 accuracy: %.3f%%, MRR accuracy: %.3f%%"
                            % (
                                i + 1,
                                batch_accuracy * 100 / (self.B * 7 + predicted.shape[0]),
                                macro_batch_accuracy * 100 / (self.B * 7 + predicted.shape[0]),
                                micro_batch_accuracy * 100 / (self.B * 7 + predicted.shape[0]),
                                p1_batch_accuracy * 100 / (self.B * 7 + predicted.shape[0]),
                                p3_batch_accuracy * 100 / (self.B * 7 + predicted.shape[0]),
                                MRR_batch_accuracy * 100 / (self.B * 7 + predicted.shape[0])
                            )
                        )
                        running_accuracy += batch_accuracy
                        macro_running_accuracy += macro_batch_accuracy
                        micro_running_accuracy += micro_batch_accuracy
                        p1_running_accuracy += p1_batch_accuracy
                        p3_running_accuracy += p3_batch_accuracy
                        MRR_running_accuracy += MRR_batch_accuracy
                        batch_accuracy = 0.0
                        macro_batch_accuracy = 0.0
                        micro_batch_accuracy = 0.0
                        p1_batch_accuracy = 0.0
                        p3_batch_accuracy = 0.0
                        MRR_batch_accuracy = 0.0

                print(
                    "[%s] f1 accuracy: %.3f%%, macro f1 accuracy: %.3f%%, micro f1 accuracy: %.3f%%, p1 accuracy: %.3f%%, p3 accuracy: %.3f%%, MRR accuracy: %.3f%%"
                    % (
                        self.data_name,
                        running_accuracy * 100 / self.test_size,
                        macro_running_accuracy * 100 / self.test_size,
                        micro_running_accuracy * 100 / self.test_size,
                        p1_running_accuracy * 100 / self.test_size,
                        p3_running_accuracy * 100 / self.test_size,
                        MRR_running_accuracy * 100 / self.test_size,
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

    
    #DBPedia dataset
    train_file = 'train-coreclass.jsonl'
    taxonomy_file = 'taxonomy-2.json'
    data_name = 'DBPEDIA'
    test_file = 'test.jsonl'
    
    """
    # amazon dataset
    train_file = "amazon-coreclass-45000.jsonl"
    taxonomy_file = "taxonomy.json"
    data_name = "amazon"
    """
    
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
    is_train = True
    is_load = False

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
        test_file= test_file
    )

    print ("Is train? : %d"%(is_train))

    trainer.prepare_train()
    if is_train : 
        if is_load:
            print("load pretrained model")
            trainer.load_pretrained_model()
        trainer.train(patience=7, threshold=0.3)
    else : 
        trainer.load_pretrained_model()
        trainer.evaluation(threshold = 0.3)

