import os
import torch
import pickle
import pandas as pd
from tqdm import tqdm
from torch.nn.functional import normalize

# 导入 ProTrek 模型
from model.ProTrek.protrek_trimodal_model import ProTrekTrimodalModel

# 指定显卡
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET = 'dualkiba-DualPGdq(work2random42)'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', DATASET))

def extract_residue_level_features(csv_path, save_path):
    # 1. 权重路径配置
    weight_dir = "/New_SSD/demo_hpf/MSN-DTA/weights/ProTrek_650M"
    config = {
        "protein_config": f"{weight_dir}/esm2_t33_650M_UR50D",
        "text_config": f"{weight_dir}/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "structure_config": f"{weight_dir}/foldseek_t30_150M",
        "load_protein_pretrained": False,
        "load_text_pretrained": False,
        "from_checkpoint": f"{weight_dir}/ProTrek_650M.pt"
    }

    # 2. 加载 ProTrek 模型
    print(f"[ProTrek] 正在加载模型到 {DEVICE}...")
    model = ProTrekTrimodalModel(**config).eval().to(DEVICE)

    # 3. 读取数据
    df = pd.read_csv(csv_path)
    seqs = df['target_sequence'].dropna().unique().tolist()
    print(f"待处理蛋白质数量: {len(seqs)}")

    # 4. 提取残基特征 (Residue-level)
    prlist = list()
    # 注意：残基特征极其占用显存，Batch Size 必须调小！
    batch_size = 1

    with torch.no_grad():
        for i in tqdm(range(0, len(seqs), batch_size), desc="提取残基级特征"):
            batch_seqs = seqs[i : i + batch_size]

            # 1. 使用模型的分词器
            inputs = model.protein_encoder.tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2000
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            # 2. 【关键修改】：显式要求输出隐藏层 (output_hidden_states=True)
            # 这样无论模型是 MaskedLM 还是基础模型，我们都能拿到特征
            outputs = model.protein_encoder.model(**inputs, output_hidden_states=True)

            # 3. 从 hidden_states 中获取最后一层特征
            # hidden_states 是一个元组，包含了从 Embedding 层到最后每一层的输出
            # 索引 -1 代表最后一层的输出 [Batch, Length, 1280]
            last_hidden_state = outputs.hidden_states[-1]

            for j, seq in enumerate(batch_seqs):
                # 计算真实长度 (去掉 padding)
                valid_len = (inputs['attention_mask'][j] == 1).sum().item()

                # 按 preparation.py 的方式保留完整有效长度，不去掉第一个和最后一个位置
                real_repr = last_hidden_state[j, :valid_len, :].cpu()
                
                prlist.append(real_repr)

            # 4. 每处理 50 条清理一次缓存
            if i % 50 == 0:
                torch.cuda.empty_cache()

    # 5. 保存为 TAPB-main/preparation.py 需要的 list + pickle 格式
    print(f"正在保存文件至 {save_path} ... (这可能需要一两分钟，因为文件很大)")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    file = open(save_path, 'wb')
    pickle.dump(prlist, file)
    print(f"✅ 处理完成！文件大小应在 GB 级别。")


if __name__ == "__main__":
    csv_file = os.path.join(DATA_PATH, 'raw', 'protein_text.csv')
    output_file = os.path.join(DATA_PATH, 'processed', 'pr_f_1280_2000.pkl')

    extract_residue_level_features(csv_file, output_file)
