import argparse
import json
import os
import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataclasses import dataclass

import pyarrow.dataset as ds

from dataset import MyTestDataset, save_emb
from model import BaselineModel


def get_ckpt_path():

    ckpt_path = os.environ.get("MODEL_OUTPUT_PATH")
    if ckpt_path is None:
        raise ValueError("MODEL_OUTPUT_PATH is not set")
    for item in os.listdir(ckpt_path):
        if item.endswith(".pt"):
            return os.path.join(ckpt_path, item)


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=101, type=int)

    # Baseline Model construction
    parser.add_argument('--hidden_units', default=32, type=int)
    parser.add_argument('--num_blocks', default=1, type=int)
    parser.add_argument('--num_epochs', default=3, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    args = parser.parse_args()

    return args


def read_result_ids(file_path):
    with open(file_path, 'rb') as f:
        # Read the header (num_points_query and FLAGS_query_ann_top_k)
        num_points_query = struct.unpack('I', f.read(4))[0]  # uint32_t -> 4 bytes
        query_ann_top_k = struct.unpack('I', f.read(4))[0]  # uint32_t -> 4 bytes

        print(f"num_points_query: {num_points_query}, query_ann_top_k: {query_ann_top_k}")

        # Calculate how many result_ids there are (num_points_query * query_ann_top_k)
        num_result_ids = num_points_query * query_ann_top_k

        # Read result_ids (uint64_t, 8 bytes per value)
        result_ids = np.fromfile(f, dtype=np.uint64, count=num_result_ids)

        return result_ids.reshape((num_points_query, query_ann_top_k))


def process_cold_start_feat(feat):
    """
    处理冷启动特征。训练集未出现过的特征value为字符串，默认转换为0.可设计替换为更好的方法。
    """
    processed_feat = {}
    for feat_id, feat_value in feat.items():
        if type(feat_value) == list:
            value_list = []
            for v in feat_value:
                if type(v) == str:
                    value_list.append(0)
                else:
                    value_list.append(v)
            processed_feat[feat_id] = value_list
        elif type(feat_value) == str:
            processed_feat[feat_id] = 0
        else:
            processed_feat[feat_id] = feat_value
    return processed_feat


def get_candidate_emb_parquet(indexer, feat_types, feat_default_value, mm_emb_dict, model):
    EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}

    candidate_path = os.path.join(os.environ.get('EVAL_DATA_PATH'), 'candidate')
    candidates = ds.dataset(str(candidate_path), format="parquet")

    item_feature_ids = ['100', '101', '102', '112', '114', '115', '116', '117', '118', '119', '120', '121', '122']

    scanner = candidates.scanner(columns=['item_id', 'retrieval_id', *item_feature_ids], batch_size=100000)

    total_rows = candidates.count_rows()
    print(f"all the candidates rows: {total_rows}")

    item_ids, record_creative_ids, record_retrieval_ids, features = [], [], [], []
    retrieve_id2creative_id = {}

    with tqdm(total=total_rows, desc="Loading Candidates", unit=" rows") as pbar:

        for batch in scanner.to_batches():
            
            item_ids_col = batch.column('item_id')
            retrieval_ids = batch.column('retrieval_id') 

            for i in range(len(batch)):
                feature = {}
                item_id_raw = item_ids_col[i].as_py()
                retrieval_id = retrieval_ids[i].as_py() 
                item_id = indexer[item_id_raw] if item_id_raw in indexer else 0

                missing_fields = set(
                    feat_types['item_sparse'] + feat_types['item_array'] + feat_types['item_continual']
                ) - set(item_feature_ids)
                for feat_id in missing_fields:
                    feature[feat_id] = feat_default_value[feat_id]
                
                # process cold start:
                for feat_id in item_feature_ids:
                    feat_id_dict = batch.column(feat_id)[i].as_py()
                    fv = feat_id_dict['feature_value']
                    try:
                        val = int(fv) if fv is not None else 0
                    except (ValueError, TypeError):
                        val = 0
                    feature[feat_id] = val

                # Clamp feature values to embedding table bounds
                for fid in feat_types.get('item_sparse', []):
                    if fid in feature and isinstance(feature[fid], int):
                        max_val = model.ITEM_SPARSE_FEAT.get(fid, 999999)
                        feature[fid] = max(0, min(feature[fid], max_val))
                for fid in feat_types.get('item_array', []):
                    if fid in feature and isinstance(feature[fid], list):
                        max_val = model.ITEM_ARRAY_FEAT.get(fid, 999999)
                        feature[fid] = [max(0, min(v, max_val)) if isinstance(v, int) else v for v in feature[fid]]

                for feat_id in feat_types['item_emb']:
                    if item_id_raw in mm_emb_dict[feat_id]:
                        feature[feat_id] = mm_emb_dict[feat_id][item_id_raw]
                    else:
                        feature[feat_id] = np.zeros(EMB_SHAPE_DICT[feat_id], dtype=np.float32)

                item_ids.append(item_id)
                record_creative_ids.append(item_id_raw)
                record_retrieval_ids.append(retrieval_id)
                features.append(feature)
                retrieve_id2creative_id[retrieval_id] = item_id_raw

            pbar.update(batch.num_rows)

    # 保存候选库的embedding和sid
    model.save_item_emb(item_ids, record_retrieval_ids, features, os.environ.get('EVAL_RESULT_PATH'))
    with open(Path(os.environ.get('EVAL_RESULT_PATH'), "retrive_id2creative_id.json"), "w") as f:
        json.dump(retrieve_id2creative_id, f)
    return retrieve_id2creative_id






def get_candidate_emb(indexer, feat_types, feat_default_value, mm_emb_dict, model):
    """
    生产候选库item的id和embedding

    Args:
        indexer: 索引字典
        feat_types: 特征类型，分为user和item的sparse, array, emb, continual类型
        feature_default_value: 特征缺省值
        mm_emb_dict: 多模态特征字典
        model: 模型
    Returns:
        retrieve_id2creative_id: 索引id->creative_id的dict
    """
    EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    candidate_path = Path(os.environ.get('EVAL_DATA_PATH'), 'predict_set.jsonl')
    # candidate_path = os.environ.get('EVAL_DATA_PATH') / 'candidate'

    # candidates = ds.dataset(str(candidate_path), format="parquet")


    item_ids, creative_ids, retrieval_ids, features = [], [], [], []
    retrieve_id2creative_id = {}

    with open(candidate_path, 'r') as f:
        for line in f:
            line = json.loads(line)
            # 读取item特征，并补充缺失值
            feature = line['features']
            creative_id = line['creative_id']
            retrieval_id = line['retrieval_id']
            item_id = indexer[item_id_raw] if item_id_raw in indexer else 0
            missing_fields = set(
                feat_types['item_sparse'] + feat_types['item_array'] + feat_types['item_continual']
            ) - set(feature.keys())
            feature = process_cold_start_feat(feature)
            for feat_id in missing_fields:
                feature[feat_id] = feat_default_value[feat_id]
            for feat_id in feat_types['item_emb']:
                if item_id_raw in mm_emb_dict[feat_id]:
                    feature[feat_id] = mm_emb_dict[feat_id][item_id_raw]
                else:
                    feature[feat_id] = np.zeros(EMB_SHAPE_DICT[feat_id], dtype=np.float32)

            item_ids.append(item_id)
            creative_ids.append(creative_id)
            retrieval_ids.append(retrieval_id)
            features.append(feature)
            retrieve_id2creative_id[retrieval_id] = item_id_raw

    # 保存候选库的embedding和sid
    model.save_item_emb(item_ids, retrieval_ids, features, os.environ.get('EVAL_RESULT_PATH'))
    with open(Path(os.environ.get('EVAL_RESULT_PATH'), "retrive_id2creative_id.json"), "w") as f:
        json.dump(retrieve_id2creative_id, f)
    return retrieve_id2creative_id


def infer():
    args = get_args()
    
    data_path = os.environ.get('EVAL_DATA_PATH')

    test_dataset = MyTestDataset(data_path, args)

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=test_dataset.collate_fn
    )

    usernum, itemnum = test_dataset.usernum, test_dataset.itemnum
    feat_statistics, feat_types = test_dataset.feat_statistics, test_dataset.feature_types
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)
    model.eval()

    ckpt_path = get_ckpt_path()
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device(args.device)))

    print("Enter get_candidate_emb_parquet")

    retrieve_id2creative_id = get_candidate_emb_parquet(
        test_dataset.indexer['i'],
        test_dataset.feature_types,
        test_dataset.feature_default_value,
        test_dataset.mm_emb_dict,
        model,
    )

    all_embs = []
    user_list = []
    for step, batch in tqdm(enumerate(test_loader), total=len(test_loader)):

        seq, token_type, seq_feat, user_id = batch
        seq = seq.to(args.device)
        logits = model.predict(seq, seq_feat, token_type)
        for i in range(logits.shape[0]):
            emb = logits[i].unsqueeze(0).detach().cpu().numpy().astype(np.float32)
            all_embs.append(emb)
        user_list += user_id

    # 生成候选库的embedding 以及 id文件
    # retrieve_id2creative_id = get_candidate_emb(
    #     test_dataset.indexer['i'],
    #     test_dataset.feature_types,
    #     test_dataset.feature_default_value,
    #     test_dataset.mm_emb_dict,
    #     model,
    # )

    

    all_embs = np.concatenate(all_embs, axis=0)
    # 保存query文件
    save_emb(all_embs, Path(os.environ.get('EVAL_RESULT_PATH'), 'query.fbin'))


    print("Enter ANN Search (Faiss Python)")

    # ---- Faiss ANN (replaces C++ binary) ----
    import faiss
    from dataset import load_fbin

    result_path = Path(os.environ.get('EVAL_RESULT_PATH'))
    item_embs, item_ids_arr = load_fbin(result_path / 'embedding.fbin'), None
    # Load item retrieval IDs
    with open(result_path / 'id.u64bin', 'rb') as f:
        n_items, _ = struct.unpack('II', f.read(8))
        item_retrieval_ids = np.fromfile(f, dtype=np.uint64, count=n_items)

    query_embs = load_fbin(result_path / 'query.fbin')
    dim = item_embs.shape[1]

    # Build HNSW index
    index = faiss.IndexHNSWFlat(dim, 64)
    index.hnsw.efConstruction = 1280
    index.hnsw.efSearch = 640
    index.add(item_embs.astype(np.float32))

    top_k = 10
    _, I = index.search(query_embs.astype(np.float32), top_k)
    # I shape: (num_queries, top_k), values are indices into item_retrieval_ids
    top10s_retrieved = item_retrieval_ids[I]  # (num_queries, top_k) retrieval IDs
    print(f"ANN search done: {I.shape}")
    top10s_untrimmed = []
    for top10 in tqdm(top10s_retrieved):
        for item in top10:
            top10s_untrimmed.append(retrieve_id2creative_id.get(int(item), 0))

    top10s = [top10s_untrimmed[i : i + 10] for i in range(0, len(top10s_untrimmed), 10)]

    return top10s, user_list
