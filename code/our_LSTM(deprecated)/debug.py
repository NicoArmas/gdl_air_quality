import torch
from airquality import AirQuality

aq = AirQuality(data_dir='../data') #data

#problemi:
# batch size diverse (quella la scartiamo)
# time window non sufficiente (forse non succede)


window = 12
batch_size = 32
n = len(aq.dataframe())
n_batches = n // (window * batch_size)
tens = None
i = 1
j = 0
batches = []



for index, row in aq.dataframe().iterrows():
    tens = torch.FloatTensor(list(row))
    tens = tens.view(-1, len(tens))
    batch = None
    while batch is None or batch.shape[-1] < batch_size:
        while i < n and i % window + 1 != 0:
            tmp = aq.dataframe().iloc[i]
            tmp = torch.FloatTensor(list(tmp))
            tmp = tmp.view(-1, len(tmp))
            tens = torch.cat([tens, tmp], 0)
            i += 1

        tens = tens.view(tens.shape[0], tens.shape[1], -1)

        if not batch:
            batch = tens
        else:
            batch = torch.cat([batch, tens], 2)
    
    batches.append(batch)
    if len(batches == n_batches):
        break
            

print(len(batches))
print(f' size ultimo batch e :{batches[-1].shape}')
print(f'size primo batch e: {batches[0].shape}')