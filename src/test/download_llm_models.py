"""
hugging 모델 다운로드 기능 
"""

#!/usr/bin/env python3
# download_llm_models.py

import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# 다운로드할 모델 리스트 (후보가 여러 개라면 여기에 추가)
MODEL_REPOS = [
    "Qwen/Qwen2.5-7B-Instruct-1M",
    # 예시: "Qwen/Qwen3-4B",
    #       "google/gemma-3n-E4B-it",
]

# 저장할 기본 경로
BASE_SAVE_DIR = "/home/조기정/project/RAG_LLM/src/test/llm_model"

def download_and_save_model(repo_id: str, save_dir: str):
    """
    repo_id Hugging Face 모델을 다운로드하여 save_dir에 저장한다.
    """
    print(f"▶ 다운로드 시작: {repo_id}")
    # 1) 토크나이저 로드 및 저장
    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
    tokenizer.save_pretrained(save_dir)
    # 2) 모델 로드 및 저장
    model = AutoModelForCausalLM.from_pretrained(repo_id, trust_remote_code=True)
    model.save_pretrained(save_dir)
    print(f"✔ 저장 완료: {save_dir}")

def main():
    os.makedirs(BASE_SAVE_DIR, exist_ok=True)

    for repo in MODEL_REPOS:
        # 모델명만 떼어내기 (슬래시 뒤 부분)
        model_name = repo.split("/")[-1]
        target_dir = os.path.join(BASE_SAVE_DIR, model_name)
        os.makedirs(target_dir, exist_ok=True)
        download_and_save_model(repo, target_dir)

if __name__ == "__main__":
    main()