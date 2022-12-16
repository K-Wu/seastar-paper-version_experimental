#!/usr/bin/env python3
# originally adapted from graphiler/python/graphiler/utils/setup.py
# get the latest version from https://github.com/K-Wu/graphiler_experimental/blob/main/python/graphiler/utils_lite/setup_lite.py
import torch

homo_dataset = {"cora": 1433, "pubmed": 500, "ppi": 50, "arxiv": 128, "reddit": 602}

hetero_dataset = ["aifb", "mutag", "bgs", "biokg", "am", "wikikg2", "mag"]


def setup(device="cuda:0"):
    torch.manual_seed(42)
    assert torch.cuda.is_available()
    device = torch.device(device)
    return device


def load_data_as_dgl_graph(name):
    import dgl
    from dgl.data.rdf import AIFB, MUTAG, BGS, AM
    from ogb.nodeproppred import DglNodePropPredDataset
    from ogb.linkproppred import DglLinkPropPredDataset

    if name == "arxiv":
        dataset = DglNodePropPredDataset(name="ogbn-arxiv")
        g = dataset[0][0]
        # isinstance(g, dgl.DGLHeteroGraph)
        # g.canonical_etypes = [('_N', '_E', '_N')]
        # g.ntypes = ['_N']
        # g.etypes = ['_E']
    elif name == "proteins":
        dataset = DglNodePropPredDataset(name="ogbn-proteins")
        g = dataset[0][0]
    elif name == "reddit":
        dataset = dgl.data.RedditDataset()
        g = dataset[0]
    elif name == "ppi":
        g = dgl.batch(
            [g for x in ["train", "test", "valid"] for g in dgl.data.PPIDataset(x)]
        )
    elif name == "cora":
        dataset = dgl.data.CoraGraphDataset()
        g = dataset[0]
    elif name == "pubmed":
        dataset = dgl.data.PubmedGraphDataset()
        g = dataset[0]
    elif name == "debug":
        g = dgl.graph(([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]))
    elif name == "aifb":
        dataset = AIFB()
        # isinstance(g, dgl.DGLHeteroGraph)
        # g.canonical_etypes = [('Forschungsgebiete', 'ontology#dealtWithIn', 'Projekte'), ('Forschungsgebiete', 'ontology#isWorkedOnBy', 'Personen'), ('Forschungsgebiete', 'ontology#name', '_Literal'), ('Forschungsgebiete', 'rdftype', '_Literal'), ...]
        # g.etypes = ['ontology#dealtWithIn', 'ontology#isWorkedOnBy', 'ontology#name', 'rdftype', 'rev-ontology#isAbout', 'rev-ontology#isAbout', 'ontology#carriesOut', 'ontology#head', 'ontology#homepage', 'ontology#member', 'ontology#name', 'ontology#publishes', 'rev-ontology#carriedOutBy', 'ontology#finances', 'ontology#name', 'rev-ontology#financedBy', 'ontology#fax', 'ontology#homepage', ...]
        # g.ntypes = ['Forschungsgebiete', 'Forschungsgruppen', 'Kooperationen', 'Personen', 'Projekte', 'Publikationen', '_Literal']
        #g = dataset[0]
        g = dataset.graph
    elif name == "mutag":
        dataset = MUTAG()
        #g = dataset[0]
        g = dataset.graph
    elif name == "bgs":
        dataset = BGS()
        #g = dataset[0]
        g = dataset.graph
    elif name == "am":
        dataset = AM()
        #g = dataset[0]
        g = dataset.graph
    elif name == "mag":
        dataset = DglNodePropPredDataset(name="ogbn-mag")
        # g.canonical_etypes = [('author', 'affiliated_with', 'institution'), ('author', 'writes', 'paper'), ('paper', 'cites', 'paper'), ('paper', 'has_topic', 'field_of_study')]
        # g.etypes = ['affiliated_with', 'writes', 'cites', 'has_topic']
        # g.ntypes = ['author', 'field_of_study', 'institution', 'paper']
        g = dataset[0][0]
    elif name == "wikikg2":
        dataset = DglLinkPropPredDataset(name="ogbl-wikikg2")
        g = dataset[0]
        src, dst = g.edges()
        reltype = torch.flatten(g.edata["reltype"]).cuda()
        num_etypes = torch.max(reltype).item() + 1
        hetero_dict = {}
        for i in range(num_etypes):
            type_index = (reltype == i).nonzero()
            hetero_dict[("n", str(i), "n")] = (
                torch.flatten(src[type_index]),
                torch.flatten(dst[type_index]),
            )
        g = dgl.heterograph(hetero_dict)
    elif name == "fb15k":
        from dgl.data import FB15k237Dataset

        dataset = FB15k237Dataset()
        g = dataset[0]
        src, dst = g.edges()
        reltype = torch.flatten(g.edata["etype"]).cuda()
        num_etypes = torch.max(reltype).item() + 1
        hetero_dict = {}
        for i in range(num_etypes):
            type_index = (reltype == i).nonzero()
            hetero_dict[("n", str(i), "n")] = (
                torch.flatten(src[type_index]),
                torch.flatten(dst[type_index]),
            )
        g = dgl.heterograph(hetero_dict)
    elif name == "biokg":
        dataset = DglLinkPropPredDataset(name="ogbl-biokg")
        g = dataset[0]
    elif name == "debug_hetero":
        g = dgl.heterograph(
            {
                ("user", "+1", "movie"): ([0, 0, 1], [0, 1, 0]),
                ("user", "-1", "movie"): ([1, 2, 2], [1, 0, 1]),
                ("user", "+1", "user"): ([0], [1]),
                ("user", "-1", "user"): ([2], [1]),
                ("movie", "+1", "movie"): ([0], [1]),
                ("movie", "-1", "movie"): ([1], [0]),
            }
        )
    else:
        raise Exception("Unknown Dataset")
    return g


if __name__ == "__main__":
    # test load dataset
    import dgl
    for dataset in hetero_dataset:
        print(dataset)
        g = load_data_as_dgl_graph(dataset)
        g = dgl.to_homo(g)
        print(g)
        print(g.ndata[dgl.NTYPE])
        print(g.ndata[dgl.NID])
        print(g.edata[dgl.ETYPE])
        print(g.edata[dgl.EID])