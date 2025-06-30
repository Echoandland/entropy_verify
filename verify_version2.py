import json
import re
import ast
import numpy as np
from collections import Counter
import requests
import argparse
from typing import List, Dict
import sys
import os
import datetime
import time
import matplotlib.pyplot as plt
from openai import OpenAI
import google.generativeai as genai





def retry_call(func, retries=3, delay=2, *args, **kwargs):
    """
    简单重试机制：调用 func，如果失败则重试若干次。
    """
    for i in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if i < retries - 1:
                time.sleep(delay)
            else:
                raise e



class Entropy_Calculation_Classify_By_GPT4:
    def __init__(self, solutions):
        self.solutions = solutions
        self.stage1_cates = ""
        self.stage2_cates = ""
        self.categories = []
        self.model = 'gpt-4o'
    
    def analyze_solutions_with_gpt4o(self, n_solutions):
        try:
            str_solutions = ''
            index = 1
            for solution in self.solutions[:n_solutions]:
                str_solutions += f"Solution{index}:" + solution + "\n"
                # str_solutions += solution + "\n"
                index += 1
            # 第一步：获取初步分组描述
            user_input = (
                "Here are several solutions to the same question:" + str_solutions +
                "Please analyze and determine how these solutions can be grouped based on the methods they use. "
                "Your classification criteria must remain strictly high-level. Place solutions in different categories only when their overarching strategies are completely distinct; differences limited to sub-steps or implementation details do not count as high-level distinctions."
                "In your response, focus on explaining your reasoning and clearly state which solution indices should be grouped together. "
                "Note that if all solutions use entirely different approaches, each should be placed in its own distinct group. "
                "In your grouping, each solution should be assigned to exactly one of the groups. Make sure to carefully check the total number of solutions"
            )
            messages = [{"role": "user", "content": user_input}]
            response = retry_call(lambda: client.chat.completions.create(model=self.model, messages=messages))
            self.categories = response.choices[0].message.content
            self.stage1_cates = self.categories

            # 第二步：提取格式化的类别分组信息
            user_input = (
                "extract the category groups from the following text: " + self.categories + 
                ' return the solution with categories like this format (for example, {1: "Solution 1, Solution 2", 2: "Solution 3, Solution 4", 3: "Solution 5"}), '
                'without any other text, and only use expressions like "Solution 1", "Solution 2"...to represent each solution, '
                'follow the example I give you. Make sure to carefully check the total number of solutions.'
            )
            messages = [{"role": "user", "content": user_input}]
            response = retry_call(lambda: client.chat.completions.create(model=self.model, messages=messages))
            self.categories = response.choices[0].message.content
            self.stage2_cates = self.categories

            # 第三步：提取最终类别列表，格式如 [1,1,2,2,3]
            user_input = (
                "extract the categories from the following text: " + self.categories + 
                'return the solution with categories like this list (for example, [1,1,2,2,3]), without any other text. '
                'Note the number of elements in the list should be exactly the same as the number of solutions. '
                'For example, in the case {1: "Solution 1, Solution 2", 2: "Solution 3, Solution 4", 3: "Solution 5"}, you should get the list [1,1,2,2,3]'
            )
            messages = [{"role": "user", "content": user_input}]
            response = retry_call(lambda: client.chat.completions.create(model=self.model, messages=messages))
            self.categories = response.choices[0].message.content
            return self.categories
        except Exception as e:
            print(f"Error in analyze_solutions_with_gpt4o: {e}")
            return "[1, 1, 1, 1, 1]"




class Entropy_Calculation_Classify_By_Gemini:
    def __init__(self, solutions, model="gemini-pro"):
        self.solutions = solutions
        self.stage1_cates = ""
        self.stage2_cates = ""
        self.categories = []
        self.model = model                  # e.g. "gemini-pro" / "gemini-1.5-pro"
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self._gen_model = genai.GenerativeModel(self.model)

    def _chat(self, prompt: str) -> str:
        rsp = self._gen_model.generate_content(prompt)
        return rsp.text.strip()

    def analyze_solutions(self, n_solutions):
        try:
            # ---- 构造题解串（与原 GPT4 类同）----
            str_solutions = ''.join(
                f"Solution{i+1}:{s}\n" for i, s in enumerate(self.solutions[:n_solutions])
            )

            # 1️⃣ 分析分组
            p1 = ("Here are several solutions to the same question:" + str_solutions +
                  "Please analyze and determine how these solutions can be grouped based on the methods they use. "
                  "Your classification criteria should be high-level. Only when solutions differ fundamentally in their overall approach should they be assigned to separate categories."
                  "In your response, focus on explaining your reasoning and clearly state which solution indices should be grouped together. "
                  "Note that if all solutions use entirely different approaches, each should be placed in its own distinct group. "
                  "In your grouping, each solution should be assigned to exactly one of the groups. Make sure to carefully check the total number of solutions")
            self.stage1_cates = self._chat(p1)

            # 2️⃣ 提出 {类别: "Solution …"} 字典
            p2 = ("extract the category groups from the following text: " + self.stage1_cates +
                  ' return the solution with categories like this format (for example, {1: "Solution 1, Solution 2", 2: "Solution 3, Solution 4", 3: "Solution 5"}), '
                'without any other text, and only use expressions like "Solution 1", "Solution 2"...to represent each solution, '
                'follow the example I give you. Make sure to carefully check the total number of solutions.')
            self.stage2_cates = self._chat(p2)

            # 3️⃣ 提出最终列表
            p3 = ("extract the categories from the following text: " + self.stage2_cates +
                  'return the solution with categories like this list (for example, [1,1,2,2,3]), without any other text. '
                'Note the number of elements in the list should be exactly the same as the number of solutions. '
                'For example, in the case {1: "Solution 1, Solution 2", 2: "Solution 3, Solution 4", 3: "Solution 5"}, you should get the list [1,1,2,2,3]')
            self.categories = self._chat(p3)
            return self.categories
        except Exception as e:
            print(f"Gemini error: {e}")
            return str([1] * n_solutions)



# def get_categories(solutions, model):
#     try:
#         entropy_calculator = Entropy_Calculation_Classify_By_GPT4(solutions)
#         entropy_calculator.model = model
#         cats = entropy_calculator.analyze_solutions_with_gpt4o(len(solutions)) 
#         return entropy_calculator.stage1_cates, entropy_calculator.stage2_cates, cats
#     except Exception as e:
#         print(f"Error in get_categgories: {e}")
#         return "", "", ""

def get_categories(solutions, model):
    try:
        if model.startswith("gemini"):
            ent = Entropy_Calculation_Classify_By_Gemini(solutions, model)
            cats = ent.analyze_solutions(len(solutions))
        else:   # 保留原 GPT-4/O 路径
            ent = Entropy_Calculation_Classify_By_GPT4(solutions)
            ent.model = model
            cats = ent.analyze_solutions_with_gpt4o(len(solutions))

        return ent.stage1_cates, ent.stage2_cates, cats
    except Exception as e:
        print(f"Error in get_categories: {e}")
        return "", "", ""


# # ----------------- 小工具 -----------------
# def safe_parse_list(text: str, expected_len: int) -> List[int]:
#     """
#     把 '[1,1,2,2,3]' 字符串解析成列表；若失败或长度不对则返回全 1 占位。
#     """
#     try:
#         lst = ast.literal_eval(text)
#         if isinstance(lst, list) and len(lst) == expected_len and all(isinstance(x, int) for x in lst):
#             return lst
#     except Exception:
#         pass
#     return [1] * expected_len

# ----------------- 核心函数 -----------------
def classify_solutions_by_models(
    solutions: List[str],
    models: List[str],
) -> Dict[str, Dict[str, str]]:
    """
    返回格式：
    {
        "o1":      {"stage1_cates": "...", "stage2_cates": "...", "cats": "..."},
        "o3":      {...},
        "gpt-4o":  {...}
    }
    """
    results = {}
    for m in models:
        s1, s2, cats = get_categories(solutions, m)
        results[m] = {                 # ★ 先建字典再赋值
            "stage1_cates": s1,
            "stage2_cates": s2,
            "cats":         cats
        }
    return results

# ----------------- CLI 主入口 -----------------
def main():
    parser = argparse.ArgumentParser(
        description="Use o1, o3, gpt-4o (or any list) to classify 10 problems' solutions."
    )
    parser.add_argument("input_json", help="包含题目与 5 解法的 JSON 文件路径")
    parser.add_argument("-m", "--models", nargs="+",
                        default=["o1", "o3", "gpt-4o"],
                        help="要调用的模型名称列表（默认：o1 o3 gpt-4o）")
    parser.add_argument("-o", "--output", default="grouped.json",
                        help="输出文件名（默认：grouped.json）")
    args = parser.parse_args()

    # 读取数据
    with open(args.input_json, "r", encoding="utf-8") as f:
        problems = json.load(f)

    overall = {}
    for idx, item in enumerate(problems, 1):
        key = f"Q{idx}"
        overall[key] = classify_solutions_by_models(item["solutions"], args.models)
        print(f"{key} done → {overall[key]}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)
    print(f"\n全部完成，结果已保存至 {args.output}")

if __name__ == "__main__":
    main()



# ------------- input demo ----------------

# [
#   {
#     "question": "……",
#     "solutions": ["解法1", "解法2", "解法3", "解法4", "解法5"]
#   },
#   … 共 10 题 …
# ]

# ------------- output demo ----------------

# {
#   "Q1": {
#     "o1": [1, 1, 2, 2, 3],
#     "o3": [1, 2, 2, 3, 3],
#     "gpt-4o": [1, 1, 1, 2, 3]
#   },
#   …
# }
