from unittest.mock import sentinel
import torch

from torch.utils import data
from transformers import AutoTokenizer, RobertaTokenizer

from .augment import Augmenter
import numpy as np

# map lm name to huggingface's pre-trained model names
lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased'}

def get_tokenizer(lm):
    if lm in lm_mp:
        return AutoTokenizer.from_pretrained(lm_mp[lm])
    else:
        return AutoTokenizer.from_pretrained(lm)


class DittoDataset(data.Dataset):
    """EM dataset"""

    def __init__(self,
                 path,
                 max_len=256,
                 size=None,
                 lm='roberta',
                 da=None,
                 kbert=False):
        self.tokenizer:RobertaTokenizer = get_tokenizer(lm)
        self.kbert = kbert
        # escape special tokens 
        self.tokenizer.add_tokens(['<head>', '</head>','<tail>','</tail>'], special_tokens=True)
        self.pairs = []
        self.labels = []
        self.max_len = max_len
        self.size = size

        if isinstance(path, list):
            lines = path
        else:
            lines = open(path, encoding='utf-8')

        for line in lines:
            s1, s2, label = line.strip().split('\t')
            self.pairs.append((s1, s2))
            self.labels.append(int(label))

        self.pairs = self.pairs[:size]
        self.labels = self.labels[:size]
        self.da = da
        if da is not None:
            self.augmenter = Augmenter()
        else:
            self.augmenter = None


    def __len__(self):
        """Return the size of the dataset."""
        return len(self.pairs)

    def __getitem__(self, idx):
        """Return a tokenized item of the dataset.

        Args:
            idx (int): the index of the item

        Returns:
            List of int: token ID's of the two entities
            List of int: token ID's of the two entities augmented (if da is set)
            int: the label of the pair (0: unmatch, 1: match)
        """
        left = self.pairs[idx][0]
        right = self.pairs[idx][1]


        if self.kbert:
            left_tokenize = self.tokenizer.tokenize(left)
            right_tokenize = self.tokenizer.tokenize(right)
            x_tokenize = [self.tokenizer.bos_token] + left_tokenize + right_tokenize + [self.tokenizer.eos_token]
            know_sent_batch, position_batch, visible_matrix_batch, seg_batch = self.add_knowledge_with_vm(x_tokenize)

            # convert tokenized x into ids
            x = self.tokenizer.convert_tokens_to_ids(know_sent_batch)
            # x_tokenize.convert_tokens_to_idx(text=left,
            #                           text_pair=right,
            #                           max_length=self.max_len,
            #                           truncation=True)
        else:
            # left + right
            x = self.tokenizer.encode(text=left,
                                    text_pair=right,
                                    max_length=self.max_len,
                                    truncation=True) # [ids, ...]
            # print(x)
            # raise Exception("debug")

        # augment if da is set
        if self.da is not None:
            combined = self.augmenter.augment_sent(left + ' [SEP] ' + right, self.da)
            left, right = combined.split(' [SEP] ')
            x_aug = self.tokenizer.encode(text=left,
                                      text_pair=right,
                                      max_length=self.max_len,
                                      truncation=True)
            return x, x_aug, self.labels[idx]
        elif self.kbert:
            return x, self.labels[idx],know_sent_batch,position_batch,visible_matrix_batch,seg_batch
        else:
            return x, self.labels[idx]
    
    def parser_adhoc(self, sent_batch):
        # input: sent_batch ['COL', 'Ġ', '<head>',"person","name","column",'</head>','Ġ',"<tail>","author", "name", "</tail>"....]
        # output: [[],[],[],["author", "name"],[],[],....]
        output = [[] for _ in range(len(sent_batch))]
        left_head_ids = [i for i, x in enumerate(sent_batch) if x == "<head>"]
        left_tail_ids = [i for i, x in enumerate(sent_batch) if x == "<tail>"]
        right_tail_ids = [i for i, x in enumerate(sent_batch) if x == "</tail>"]
        assert len(left_tail_ids) == len(right_tail_ids)
        for i in range(len(left_tail_ids)):
            fir_tail = left_tail_ids[i]
            sec_tail = right_tail_ids[i]
            tail_value = sent_batch[fir_tail:sec_tail+1]
            output[left_head_ids[i]+1] = tail_value
        return output
    
    def add_knowledge_with_vm(self, sent_batch, max_entities=128, add_pad=True, max_length=128):
        """
        input: sent_batch - list of sentences, e.g., ["abcd", "efgh"]
        return: know_sent_batch - list of sentences with entites embedding
                position_batch - list of position index of each character.
                visible_matrix_batch - list of visible matrixs
                seg_batch - list of segment tags
        """
        # know_sent_batch = []
        # position_batch = []
        # visible_matrix_batch = []
        # seg_batch = []

        # create tree
        sent_tree = []
        pos_idx_tree = []
        abs_idx_tree = []
        pos_idx = -1
        abs_idx = -1
        abs_idx_src = []
        # skip_idx = []
        head_left_id, head_right_id = -1, -1
        tail_left_id, tail_right_id = -1, -1
        # print('sent_batch', sent_batch) 
        for i,token in enumerate(sent_batch): 
            # token_skip_idx = []
            # skip = False
            if token=='<head>':
                # skip_idx.append(i)
                # token_skip_idx.append(i)
                # skip = True
                head_left_id = i
                head_right_id = sent_batch[i:].index('</head>') + i
                tail_left_id = sent_batch[i:].index("<tail>")+1 +i
                tail_right_id = sent_batch[i:].index("</tail>") + i
                
                # tail_skip_idx += list(range(i+tail_left_id-1,tail_right_id+i+1))
                # token_skip_idx += [i+tail_left_id-1,tail_right_id+i]
                # skip = True
                # entities= [" ".join(tail_value)] # entities: ["author name"]
                entities = [] 
                # print(entities)
                # raise NotImplementedError
            # elif token == '</head>':
                # skip_idx.append(i)
                # token_skip_idx.append(i)
                # skip = True
            elif head_left_id < i < head_right_id:
                tail_value = sent_batch[tail_left_id:tail_right_id]
                entities = tail_value
            elif tail_left_id <= i <= tail_right_id:
                continue
            else: entities = []

            # if not skip:
            if  token not in ['<head>','</head>','<tail>','</tail>']: # and not (tail_left_id <= i <= tail_right_id):
                sent_tree.append((token, entities))
                token_pos_idx = [pos_idx+1] #[pos_idx+i for i in range(1, len(token)+1)]
                token_abs_idx = [abs_idx+1] #[abs_idx+i for i in range(1, len(token)+1)]
                abs_idx = token_abs_idx[-1]

            entities_pos_idx = []
            entities_abs_idx = []
            # for ent in entities:
            # if len(entities) >0:
            for ent in entities:
                if ent not in ['<tail>','</tail>']:
                    ent_pos_idx = [token_pos_idx[-1]+1]#[token_pos_idx[-1] + p for p in range(1, len(entities)+1)]#ent
                    entities_pos_idx.append(ent_pos_idx)
                    ent_abs_idx = [abs_idx + 1]#[abs_idx + a for a in range(1, len(entities)+1)]#ent
                    abs_idx = ent_abs_idx[-1]
                    entities_abs_idx.append(ent_abs_idx)
            if  token not in ['<head>','</head>','<tail>','</tail>']:
                pos_idx_tree.append((token_pos_idx, entities_pos_idx))
                abs_idx_tree.append((token_abs_idx, entities_abs_idx))
                pos_idx = token_pos_idx[-1]
                abs_idx_src += token_abs_idx

        # Get know_sent and pos
        know_sent = []
        pos = []
        seg = []
        # print('sent_tree',sent_tree)
        for i in range(len(sent_tree)):
            word = sent_tree[i][0]
            # if word in self.special_tags:
            #     know_sent += [word]
            #     seg += [0]
            # else:
            # print(word, i)
            # if i not in skip_idx:
            add_word = [word]#self.tokenizer.tokenize(word)
            know_sent += add_word 
            seg += [0] * len(add_word)
            pos += pos_idx_tree[i][0]

            for j in range(len(sent_tree[i][1])):
                add_word = [sent_tree[i][1][j]]
                know_sent += add_word
                seg += [1] #* len(add_word)
                pos += pos_idx_tree[i][1][j]

        token_num = len(know_sent)
        # assert token_num == abs_idx_tree, "length of know_sent = abs_idx_tree maximum idx"
        # Calculate visible matrix
        visible_matrix = np.zeros((token_num, token_num))
        # print(visible_matrix.shape)
        # print(abs_idx_tree)
        for item in abs_idx_tree:
            src_ids = item[0]
            for id in src_ids:
                visible_abs_idx = abs_idx_src + [idx for ent in item[1] for idx in ent]
                visible_matrix[id, visible_abs_idx] = 1
            for ent in item[1]:
                for id in ent:
                    visible_abs_idx = ent + src_ids
                    visible_matrix[id, visible_abs_idx] = 1

        src_length = len(know_sent)
        if len(know_sent) < max_length:
            pad_num = max_length - src_length
            know_sent += [self.tokenizer.pad_token] * pad_num # TODO
            seg += [0] * pad_num
            pos += [max_length - 1] * pad_num
            visible_matrix = np.pad(visible_matrix, ((0, pad_num), (0, pad_num)), 'constant')  # pad 0
        else:
            know_sent = know_sent[:max_length]
            seg = seg[:max_length]
            pos = pos[:max_length]
            visible_matrix = visible_matrix[:max_length, :max_length]
        
        # know_sent_batch.append(know_sent)
        # position_batch.append(pos)
        # visible_matrix_batch.append(visible_matrix)
        # seg_batch.append(seg)

    
        return know_sent, pos, visible_matrix, seg

    # @staticmethod
    def pad(self, batch):
        """Merge a list of dataset items into a train/test batch
        Args:
            batch (list of tuple): a list of dataset items

        Returns:
            LongTensor: x1 of shape (batch_size, seq_len)
            LongTensor: x2 of shape (batch_size, seq_len).
                        Elements of x1 and x2 are padded to the same length
            LongTensor: a batch of labels, (batch_size,)
        """
        if len(batch[0]) == 3:
            x1, x2, y = zip(*batch)

            maxlen = max([len(x) for x in x1+x2])
            x1 = [xi + [0]*(maxlen - len(xi)) for xi in x1]
            x2 = [xi + [0]*(maxlen - len(xi)) for xi in x2]
            return torch.LongTensor(x1), \
                   torch.LongTensor(x2), \
                   torch.LongTensor(y)
        elif len(batch[0]) == 6:
            #  know_sent, pos, visible_matrix, seg
            x1, y, x2, x3, x4, x5 = zip(*batch)

            # maxlen = max([len(x) for x in x1+x3+x4])
            # x1 = [xi + [0]*(maxlen - len(xi)) for xi in x1]
            # x2 = [xi + [0]*(maxlen - len(xi)) for xi in x2]
            # x3 = [xi + [maxlen-1]*(maxlen - len(xi)) for xi in x3]
            # x3 = [a + [0]*(maxlen - len(xi)) for xi in x3 for a in xi]
            # x4 = [xi + [0]*(maxlen - len(xi)) for xi in x4]
            # x5 = [xi + [0]*(maxlen - len(xi)) for xi in x5]
            return  torch.LongTensor(x1), \
                    torch.LongTensor(x3), \
                    torch.LongTensor(x4), \
                    torch.LongTensor(y)
        else:
            x12, y = zip(*batch)
            maxlen = max([len(x) for x in x12])
            x12 = [xi + [0]*(maxlen - len(xi)) for xi in x12]
            return torch.LongTensor(x12), \
                   torch.LongTensor(y)

