from pydantic import BaseModel
from typing import List, Dict, Any
import spacy
import json
import re
from collections import defaultdict, Counter
import os
import difflib
from difflib import get_close_matches
import requests


# spaCy 모델 로드
nlp = spacy.load("en_core_web_lg")

resource_dir = os.getenv('RESOURCE_DIR', 'v1/resources')

# JSON 데이터 로드 (단어 레벨용)
with open(f'{resource_dir}/resources/final_corrected_combined_word_levels.json', 'r', encoding='utf-8') as f:
    word_levels = json.load(f)

# TXT 데이터 로드 (워드 패밀리 체크용)
def load_word_families(file):
    word_families = []
    with open(f'{resource_dir}/resources/{file}', 'r') as f:
        for line in f:
            word_families.append(set(line.strip().split()))
    return word_families

word_families = load_word_families('merged_word_family_no_duplicates.txt')

# 요청 데이터 모델 정의
class PosTypesRequest(BaseModel):
    sentences: str


# pos 유형빈도 판단
async def get_pos_types(request: PosTypesRequest): 
    """
    입력된 문장에서 사용된 품사의 종류와 수를 반환하는 함수
    :param sentences: 분석할 문장들 (str)
    :return: 각 문장의 단어별 품사 정보와 각 문장의 품사 수
    """
    sentences = re.split(r'[.!?]', request.sentences)  # 문장을 문단으로 분리
    data = {}

    pos_count = {}  # 모든 문장의 품사를 카운트하기 위한 딕셔너리

    for index, sentence in enumerate(sentences):
        sentence = sentence.strip()  # 문장 양 끝의 공백 제거
        if not sentence:  # 빈 문장은 무시
            continue
        doc = nlp(sentence)
        pos_info = {token.text: token.pos_ for token in doc}
        sentence_key = f"{index}"
        data[sentence_key] = pos_info

        # 각 품사의 빈도를 전체적으로 계산
        for pos in pos_info.values():
            if pos in pos_count:
                pos_count[pos] += 1
            else:
                pos_count[pos] = 1

    return {"data": data, "result": pos_count}
