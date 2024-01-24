import torch

if datasets_name == "mgtab":
    # select the edge type of following and friends in MGTAB
    edge_index = torch.load("Dataset/MGTAB/edge_index.pt").T.tolist()
    edge_type = torch.load("Dataset/MGTAB/edge_type.pt").tolist()

    edge_type_01 = []
    edge_index_01 = []
    edge_index_0 = []
    edge_index_1 = []
    for i in range(edge_type.shape[0]):
        if edge_type[i] == 0:
            edge_index_01.append(edge_index[i])
            edge_type_01.append(0)
            edge_index_0.append(edge_index[i])
        if edge_type[i] == 1:
            edge_index_01.append(edge_index[i])
            edge_type_01.append(1)
            edge_index_1.append(edge_index[i])

    edge_index_01 = torch.tensor(edge_index_01).T
    edge_type_01 = torch.tensor(edge_type_01)
    edge_index_0 = torch.tensor(edge_index_0).T
    edge_index_1 = torch.tensor(edge_index_1).T
    torch.save(edge_type_01,"Dataset/MGTAB/edge_type_01.pt")
    torch.save(edge_index_01,"Dataset/MGTAB/edge_index_01.pt")
    torch.save(edge_index_0,"Dataset/MGTAB/edge_index_0.pt")
    torch.save(edge_index_1,"Dataset/MGTAB/edge_index_1.pt")

if datasets_name == "twibot-20":
    edge_index = torch.load("Dataset/Twibot-20/edge_index.pt").T.tolist()
    edge_type = torch.load("Dataset/Twibot-20/edge_type.pt").tolist()

    edge_index_0 = []
    edge_index_1 = []
    for i in range(edge_type.shape[0]):
        if edge_type[i] == 0:
            edge_index_0.append(edge_index[i])
            
        if edge_type[i] == 1:
            edge_index_1.append(edge_index[i])
            

    edge_index_0 = torch.tensor(edge_index_0).T
    edge_index_1 = torch.tensor(edge_index_1).T

    torch.save(edge_index_0,"Dataset/Twibot-20/edge_index_0.pt")
    torch.save(edge_index_1,"Dataset/Twibot-20/edge_index_1.pt")

if datasets_name == "cresci-15":
    edge_index = torch.load("Dataset/Cresci-15/edge_index.pt").T.tolist()
    edge_type = torch.load("Dataset/Cresci-15/edge_type.pt").tolist()

    edge_index_0 = []
    edge_index_1 = []
    for i in range(edge_type.shape[0]):
        if edge_type[i] == 0:
            edge_index_0.append(edge_index[i])
            
        if edge_type[i] == 1:
            edge_index_1.append(edge_index[i])
            

    edge_index_0 = torch.tensor(edge_index_0).T
    edge_index_1 = torch.tensor(edge_index_1).T

    torch.save(edge_index_0,"Dataset/Cresci-15/edge_index_0.pt")
    torch.save(edge_index_1,"Dataset/Cresci-15/edge_index_1.pt")