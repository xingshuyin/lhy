from transformers import BertTokenizer, BertModel, AdamW
from datasets import load_dataset
import torch.nn as nn
import torch

token = BertTokenizer.from_pretrained('bert-base-chinese')
pmodel = BertModel.from_pretrained('bert-base-chinese')


class Data(torch.utils.data.Dataset):
    def __init__(self, split):
        self.d = load_dataset('lansinuote/ChnSentiCorp', split=split)

    def __len__(self):
        return len(self.d)

    def __getitem__(self, i):
        return self.d[i]['text'], self.d[i]['label']


data_train = Data('train')


def collate_fn(data):  # 自定义batch数据的输出格式, data就是生成的一个batch
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]
    #编码
    data = token.batch_encode_plus(batch_text_or_text_pairs=sents, truncation=True, padding='max_length', max_length=500, return_tensors='pt', return_length=True)
    #input_ids:编码之后的数字
    #attention_mask:是补零的位置是0,其他位置是1
    return {'input_ids': data['input_ids'], 'attention_mask': data['attention_mask'], 'token_type_ids': data['token_type_ids'], 'labels': torch.tensor(labels)}
    # return data['input_ids'], data['attention_mask'], data['token_type_ids'], torch.tensor(labels)


loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=16, shuffle=True, drop_last=True, collate_fn=collate_fn)

# class Model(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc = nn.Linear(768, 2)

#     def forward(self, input_ids, attention_mask, token_type_ids):
#         with torch.no_grad():
#             input_ids, attention_mask, token_type_ids = input_ids.to('cuda'), attention_mask.to('cuda'), token_type_ids.to('cuda')
#             out = pretrained(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

#         out = self.fc(out.last_hidden_state[:, 0])

#         out = out.softmax(dim=1)

#         return out


class MyModule(nn.Module):
    def __init__(self) -> None:
        super(MyModule, self).__init__()
        self.net = nn.Sequential(nn.Linear(768, 2), )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        # print(1,x.size())
        with torch.no_grad():
            t = pmodel(**x)
        x = self.net(t.last_hidden_state[:, 0])
        x.softmax(dim=1)
        # print(2,x.size())
        # x = x.squeeze(1)  # (B, 1) -> (B)
        return x

    def calc_loss(self, pre, tar):  # 计算loss
        return self.loss(pre, tar)


model = MyModule()
optimizer = AdamW(model.parameters(), lr=5e-4)
for index, i in enumerate(loader_train):
    optimizer.zero_grad()
    labels =  i.pop('labels')
    r = model(i)
    loss = model.calc_loss(r, labels)
    loss.backward()
    optimizer.step()
    print(r.argmax(dim=1))
    # if index % 5 == 0:
    #     out = out.argmax(dim=1)
    #     accuracy = (out == i['label']).sum().item() / len(i['label'])

    #     print(i, loss.item(), accuracy)

    if index == 300:
        break