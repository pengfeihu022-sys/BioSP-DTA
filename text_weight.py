import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.nn.functional import normalize

# 导入 ProTrek 模型 (请确保你的运行路径下能正确找到 model 文件夹)
from model.ProTrek.protrek_trimodal_model import ProTrekTrimodalModel

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
DATASET = 'dualdavis-DualPGdq(work2)'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', DATASET))

def main():
    # 1. 路径设置
    protein_db_csv = os.path.join(DATA_PATH, 'raw', 'protein_text.csv')
    #output_file = "/New_SSD/demo_hpf/MSN-DTA/weights/prot_text_graphs.pt" 
    output_file = os.path.join(DATA_PATH, 'processed', 'prot_text_graphs.pt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 【重要修改】：配置 ProTrek 权重路径 (请根据你服务器上实际的权重存放路径进行修改，建议用绝对路径)
    weight_dir = "/New_SSD/demo_hpf/MSN-DTA/weights/ProTrek_650M" # 修改为你的实际路径
    config = {
        "protein_config": f"{weight_dir}/esm2_t33_650M_UR50D",
        "text_config": f"{weight_dir}/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "structure_config": f"{weight_dir}/foldseek_t30_150M",
        "load_protein_pretrained": False,
        "load_text_pretrained": False,
        "from_checkpoint": f"{weight_dir}/ProTrek_650M.pt"
    }

    # 2. 读取蛋白质文件
    print(f"正在读取蛋白质数据库: {protein_db_csv}")
    df = pd.read_csv(protein_db_csv)
    
    # 去重处理
    df = df.drop_duplicates(subset=['target_sequence']).reset_index(drop=True)
    sequences = df['target_sequence'].astype(str).tolist()
    texts = df['target_text'].fillna("").astype(str).tolist()
    print(f"总计找到 {len(sequences)} 个独立蛋白质。")

    # 3. 加载 ProTrek 模型
    print("正在加载 ProTrek 预训练模型...")
    model = ProTrekTrimodalModel(**config).eval().to(device)

    # 4. 分批提取特征 (Batch 提取)
    batch_size = 16
    all_embeddings = []
    
    print("开始提取文本特征...")
    for i in tqdm(range(0, len(texts), batch_size), desc="提取文本特征"):
        batch_texts = texts[i: i + batch_size]
        with torch.no_grad():
            # ProTrek 的 get_text_repr 自动处理分词和前向传播
            batch_repr = model.get_text_repr(batch_texts)
        all_embeddings.append(batch_repr.cpu())

    # 拼接所有批次的特征
    embeddings = torch.cat(all_embeddings, dim=0)
    
    # 归一化 (多模态对比学习的模型一般提取出来的特征需要 L2 归一化，保持你原代码的习惯)
    embeddings = normalize(embeddings, dim=-1)

    # 5. 存入字典，Key是sequence，Value是Tensor
    text_dict = {}
    for i, seq in enumerate(sequences):
        text_dict[seq] = embeddings[i].squeeze()

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    torch.save(text_dict, output_file)
    print(f"🎉 提取完成！特征字典已保存至: {output_file}")
    print(f"提取的特征维度为: {embeddings.shape[1]}")

if __name__ == "__main__":
    main()
