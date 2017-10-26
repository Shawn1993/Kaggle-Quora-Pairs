import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np
from datetime import datetime
from dl.vocab import PAD_INDEX
from utils import pkl_utils


class ModelBase(nn.Module):

    def __init__(self, prefix='', gpu=-1, batch_size=64):
        super(ModelBase, self).__init__()
        self.prefix = prefix
        self._best_score = float('inf')

        self.batch_size = batch_size
        self.gpu = gpu if torch.cuda.is_available() else -1
        if gpu >= 0:
            self.LongTensor = torch.cuda.LongTensor
            self.FloatTensor = torch.cuda.FloatTensor
        else :
            self.LongTensor = torch.LongTensor
            self.FloatTensor = torch.FloatTensor

    def _loss(self, logits, labels):
        return F.cross_entropy(logits, labels.view(-1).long())

    def _eval(self, logits, labels):
        evaluates = (F.softmax(logits).max(1)[1].long() == labels.long()).long()
        return torch.sum(evaluates)

    def _predict(self, logits):
        return F.softmax(logits)[:, 1:].data.cpu().numpy()

    def _load_batch(self, data, training):
        question1, question2, labels, size = data['question1'], data['question2'], data.get('is_duplicate'), len(data)
        leak = data[['magic_max-freq','magic_min-freq', 'magic_intersect']].values
        start = 0
        while True:
            end = start + self.batch_size
            q1_maxlen = max([len(x) for x in question1[start:end]])
            q2_maxlen = max([len(x) for x in question2[start:end]])
            question1_batch = [x + [PAD_INDEX]*(q1_maxlen-len(x)) for x in question1[start:end]]
            question2_batch = [x + [PAD_INDEX]*(q2_maxlen-len(x)) for x in question2[start:end]]
            leak_batch = leak[start:end]
            if labels is not None:
                labels_batch = labels[start:end].as_matrix()
                yield Variable(self.LongTensor(question1_batch), volatile=not training),\
                      Variable(self.LongTensor(question2_batch), volatile=not training),\
                      Variable(self.FloatTensor(leak_batch), volatile=not training),\
                      Variable(torch.from_numpy(labels_batch).type(self.FloatTensor).unsqueeze(1), volatile=not training)
            else :
                yield Variable(self.LongTensor(question1_batch), volatile=not training),\
                      Variable(self.LongTensor(question2_batch), volatile=not training),\
                      Variable(self.FloatTensor(leak_batch), volatile=not training)
            start = end
            if start > size:
                break

    def train(self, train_data, dev_data=None, max_iter=2, shuffle=True):
        if self.gpu >= 0 :
            torch.cuda.set_device(self.gpu)
            self.cuda()
        optimizer = Adam(self.parameters())
        steps = 1
        eval_loss = None
        for itr in range(max_iter):
            # shuffle
            if shuffle:
                train_data = train_data.sample(frac=1)

            start_time = datetime.now()
            all_loss = 0
            data_size = 0

            for q1, q2, leak, labels in self._load_batch(train_data, True):
                logits = self(q1, q2, leak)
                loss = self._loss(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # for logging
                sampel_size = len(labels)
                data_size += sampel_size
                all_loss += loss.data[0] * sampel_size
                steps += 1
                mean_loss = all_loss / data_size
                sys.stdout.write('\rIter[{}] - Step[{}] - Loss:{:.6f} - Timer:{}'.format(itr + 1,
                                                                                         steps,
                                                                                         loss.data[0],
                                                                                         datetime.now() - start_time))
                # if steps % 400 == 0 and itr>=0 and dev_data is not None:
                #     sys.stdout.write('\n')
                #     self.eval(dev_data, 'Dev', steps)

            sys.stdout.write('\n')
            # if (itr+1) % 1 == 0 and dev_data is not None:
            #     eval_loss =  self.eval(dev_data, 'Dev', steps)
        if dev_data is not None:
            return self.eval(dev_data, 'Dev', steps)

    def eval(self, data, name, steps, save=True):
        all_loss = 0
        all_correct = 0
        for q1, q2, leak, labels in self._load_batch(data, False):
            logits = self(q1, q2, leak)
            loss = self._loss(logits, labels)
            correct = self._eval(logits, labels)

            all_loss += loss.data[0] * len(labels)
            all_correct += correct.data[0]

        data_size = len(data)
        mean_loss = all_loss / data_size
        mean_correct = all_correct / data_size

        sys.stdout.write('[{}] - Acc:{:.4f}({}/{}) - Loss:{:.6f}\n'.format(name,
                                                                           mean_correct,
                                                                           all_correct,
                                                                           data_size,
                                                                           mean_loss))
        return mean_loss
        # if mean_loss < self._best_score and save and name=='Dev':
        #     self._best_score = mean_loss
        #     self.save(self.prefix)
        #     sys.stdout.write('save as the best!\n')


    def predict(self, data):
        if self.gpu >= 0 :
            torch.cuda.set_device(self.gpu)
            self.cuda()
        all_predicts = np.empty([0, 1])
        for q1, q2, leak in self._load_batch(data, False):
            cur_predicts = self._predict(self(q1, q2, leak))
            all_predicts = np.concatenate([all_predicts, cur_predicts])
        return all_predicts

    def save(self, name):
        torch.save(self.state_dict(), name)
        torch.save(self.config, name + '.config')

    @classmethod
    def load(clazz, name):
        config = torch.load(name+'.config')
        obj = clazz(**config)
        obj.load_state_dict(torch.load(name))
        return obj

class SelfAttentionBiLSTM(ModelBase):

    def __init__(self, vocab, prefix='', gpu=-1, batch_size=64):
        super(SelfAttentionBiLSTM, self).__init__(prefix, gpu, batch_size)
        vocab_size = len(vocab)
        vector_size = vocab.vector_size
        vectors = vocab.vectors
        
        local = locals()
        self.config = {k: local[k] for k in ('vocab_size', 'vector_size', 'prefix', 'gpu', 'batch_size')}

        V = vocab_size
        D = vector_size 
        H = 100
        da = 100
        r = 100
        H1 = 100
        H2 = 500

        self.embed = nn.Embedding(V, D)

        self.lstm = nn.LSTM(D, H, 1, batch_first=True, bidirectional=True)
        
        self.att_fc1 = nn.Linear(2*H, da, bias=False)
        self.att_fc2 = nn.Linear(da, r, bias=False)

        self.mul_fc = nn.Linear(2*r*H, H1)
        self.max_fc = nn.Linear(2*r*H, H1)
        self.sub_fc = nn.Linear(2*r*H, H1)
        self.rep_fc = nn.Linear(2*r*H, H1)
        self.leak_rep_fc = nn.Linear(3, H1)

        self.fc1 = nn.Linear(6*H1, H2)
        self.output = nn.Linear(H2, 2)

        if vectors is not None:
            self.embed.weight.data = torch.from_numpy(vectors).type_as(self.embed.weight.data)
        self._init_weights()


    def _init_weights(self):
        initrange = 0.001
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.fill_(0)
        self.output.weight.data.uniform_(-initrange, initrange)
        self.output.bias.data.fill_(0)
        self.att_fc1.weight.data.uniform_(-initrange, initrange)
        self.att_fc2.weight.data.uniform_(-initrange, initrange)


    def forward(self, x, y, leak):
        x_mask = (x != PAD_INDEX).float().unsqueeze(2) # (N, W1, 1)
        y_mask = (y != PAD_INDEX).float().unsqueeze(2) # (N, W2, 1)

        # Embbeding
        x = self.embed(x)  # (N, W1, D)
        y = self.embed(y)  # (N, W2, D)

        # BiLSTM
        x_out, _ = self.lstm(x) # (N, W1, 2*H)
        y_out, _ = self.lstm(y) # (N, W2, 2*H)
        x_out = x_out * x_mask.expand_as(x_out) # (N, W1, 2*H)
        y_out = y_out * y_mask.expand_as(y_out) # (N, W2, 2*H)

        # self Attention
        a_x = F.tanh(self.att_fc1(x_out.view(-1, x_out.size(2)))) # (N*W1, da)
        a_x = self.att_fc2(a_x) # (N*W1, r)
        a_x = a_x.view(-1, x_out.size(1), a_x.size(1)).transpose(1,2).contiguous() # (N, r, W1)
        a_x = F.softmax(a_x.view(-1, a_x.size(2))).view_as(a_x) # (N, r, W1)
        x_represent = torch.bmm(a_x, x_out) # (N, r, 2*H)
        x_represent = F.relu(x_represent.view(x_represent.size(0), -1)) # (N, r*2*H)

        a_y = F.tanh(self.att_fc1(y_out.view(-1, y_out.size(2)))) # (N*W2, da)
        a_y = self.att_fc2(a_y) # (N*W2, r)
        a_y = a_y.view(-1, y_out.size(1), a_y.size(1)).transpose(1,2).contiguous() # (N, r, W2)
        a_y = F.softmax(a_y.view(-1, a_y.size(2))).view_as(a_y) # (N, r, W2)
        y_represent = torch.bmm(a_y, y_out) # (N, r, 2*H)
        y_represent = F.relu(y_represent.view(y_represent.size(0), -1)) # (N, r*2*H)

        # Matching
        max_rep = self.max_fc(torch.max(x_represent, y_represent)) # (N, 100)
        sub_rep = self.sub_fc(torch.abs(x_represent-y_represent)) # (N, 100)
        mul_rep = self.mul_fc(x_represent*y_represent) # (N, 100)
        x_rep = self.rep_fc(x_represent) # (N, 100)
        y_rep = self.rep_fc(y_represent) # (N, 100)

        leak_rep = self.leak_rep_fc(leak)
        all_represent = torch.cat([ mul_rep, 
                                    sub_rep,
                                    max_rep,
                                    x_rep, 
                                    y_rep,
                                    leak_rep
                                    ], 1) # (N, 500)
        z = F.relu(all_represent) # (N, 500)
        z = F.relu(self.fc1(z))

        return self.output(z)

class SimpleCNN(ModelBase):
    def __init__(self, vocab, prefix='', gpu=-1, batch_size=64):
        super(SimpleCNN, self).__init__(prefix, gpu, batch_size)
        vocab_size = len(vocab)
        vector_size = vocab.vector_size
        vectors = vocab.vectors
        
        local = locals()
        self.config = {k: local[k] for k in ('vocab_size', 'vector_size', 'prefix', 'gpu', 'batch_size')}

        V = vocab_size
        D = vector_size
        Co = 128

        self.Ks = [3, 4, 5]
        H1 = 100
        H2 = 500

        self.embed = nn.Embedding(V, D)
        self.convs = nn.ModuleList([nn.Conv2d(1, Co, (K, D)) for K in self.Ks])

        self.mul_fc = nn.Linear(len(self.Ks)*Co, H1)
        self.max_fc = nn.Linear(len(self.Ks)*Co, H1)
        self.sub_fc = nn.Linear(len(self.Ks)*Co, H1)
        self.rep_fc = nn.Linear(len(self.Ks)*Co, H1)
        self.leak_rep_fc = nn.Linear(3, H1)

        self.fc1 = nn.Linear(6*H1, H2)
        self.output = nn.Linear(H2,2)
        
        if vectors is not None:
            self.embed.weight.data = torch.from_numpy(vectors).type_as(self.embed.weight.data)
        self._init_weights()


    def _init_weights(self):
        initrange = 0.001
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.fill_(0)
        self.output.weight.data.uniform_(-initrange, initrange)
        self.output.bias.data.fill_(0)

    def forward(self, x, y, leak):

        max_ks = max(self.Ks)
        if x.size(1)<max_ks:
            x.data = torch.cat([x.data, self.LongTensor(x.size(0), max_ks-x.size(1)).fill_(PAD_INDEX)], 1)
        if y.size(1)<max_ks:
            y.data = torch.cat([y.data, self.LongTensor(x.size(0), max_ks-y.size(1)).fill_(PAD_INDEX)], 1)

        x_mask = (x != PAD_INDEX).float().unsqueeze(2) # (N, W1, 1)
        y_mask = (y != PAD_INDEX).float().unsqueeze(2) # (N, W2, 1)

        # Embbeding
        x = self.embed(x)  # (N, W1, D)
        y = self.embed(y)  # (N, W2, D)

        # Mask
        x = x * x_mask.expand_as(x) # (N, W1, D)
        y = y * y_mask.expand_as(y) # (N, W2, D)

        x = x.unsqueeze(1) # (N, 1, W1, D)
        y = y.unsqueeze(1) # (N, 1, W2, D)


        x_outs = [conv(x).squeeze(3) for conv in self.convs] # (N, Co, W1)
        y_outs = [conv(y).squeeze(3) for conv in self.convs] # (N, Co, W2)

        # print(x_outs[0].size())
        x_outs = [F.max_pool1d(F.relu(x), x.size(2)).squeeze(2) for x in x_outs] # (N, Co)
        y_outs = [F.max_pool1d(F.relu(y), y.size(2)).squeeze(2) for y in y_outs] # (N, Co)

        x_represent = torch.cat(x_outs, 1) # (N, 3*Co)
        y_represent = torch.cat(y_outs, 1) # (N, 3*Co)

        # Matching
        max_rep = self.max_fc(torch.max(x_represent, y_represent)) # (N, H1)
        sub_rep = self.sub_fc(torch.abs(x_represent-y_represent)) # (N, H1)
        mul_rep = self.mul_fc(x_represent*y_represent) # (N, H1)
        x_rep = self.rep_fc(x_represent) # (N, H1)
        y_rep = self.rep_fc(y_represent) # (N, H1)

        leak_rep = self.leak_rep_fc(leak)
        all_represent = torch.cat([mul_rep,
                                   sub_rep,
                                   max_rep,
                                   x_rep,
                                   y_rep,
                                   leak_rep,
                                   ], 1)  # (N, 6*H1)


        z = F.relu(all_represent) # (N, 6*H1)
        z = F.relu(self.fc1(z)) # (N, H2) 

        return self.output(z) # (N, 2) 