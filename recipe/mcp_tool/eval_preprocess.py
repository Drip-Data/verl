import json
import os

import datasets
import pandas as pd
from datasets import concatenate_datasets
from omegaconf import OmegaConf

if __name__ == "__main__":
    # tools_config_file = "/workspace/project/verl/recipe/mcp_tool/tool_schema.yaml"
    tools_config_file = "./tool_schema.yaml"
    tools_config = OmegaConf.load(tools_config_file)
    # 创建一个空列表来存放所有的 tool_schema
    all_tool_schemas = []
    tools_kwargs = {}
    tools_names = []

    # 遍历 tools 列表中的每一个工具
    for tool_item in tools_config["tools"]:
        # 将当前的 tool_schema 转换为 Python 字典
        tool_schema = OmegaConf.to_container(tool_item)
        # 将转换后的字典添加到列表中
        all_tool_schemas.append(tool_schema)
        tools_names.append(tool_schema["function"]["name"])
        tools_kwargs[tool_schema["function"]["name"]] = {
            "create_kwargs": {"dummy": "dummy"},
        }

    # 最终将包含所有 schema 的列表转换为 JSON 字符串
    tools_json_string = json.dumps(all_tool_schemas, indent=2)
    tools_names_string = ",".join(tools_names)

    # print(type(tools_json_string))

    math_data_file = '/workspace/verl/data/eval/math_eval.json'
    search_data_file = '/workspace/verl/data/eval/search_eval.json'

    math_data = datasets.load_dataset('json', data_files=math_data_file)
    search_data = datasets.load_dataset('json', data_files=search_data_file)

    math_data = math_data['train']
    search_data = search_data['train']

    system_prompt = f"""
You are a helpful assistant that can solve the given question step by step with the help of multiple tools. Given a question, you need to first think about the reasoning process in the mind and then provide the answer. During thinking, you can invoke the following tools, with parameter schema:

{tools_names_string}

The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively, and the search query and result are enclosed within <tool_call> </tool_call> and <result> </result> tags respectively. For example,

1. For thinking: ''' <think>your thinking process</think> ''' 
2. For tool call: ''' <tool_call> {{
  "name": "tool_name",
  "arguments": {{
      "param": "val", ...
  }}
}}</tool_call> ''' 
3. For final answer: ''' <answer>\\boxed{{Your final answer here}}</answer> ''' 
"""

    

    def make_map_fn():
        def process_fn(example, idx):
            question_raw = example.pop("query")

            question = question_raw

            answer_raw = example.pop("answer")
            
            data = {
                "data_source":"mcp_tools1",
                "agent_name": "tool_agent",
                "prompt": [
                    {
                        "role": "system",
                        "content": 
                            system_prompt
                        ,
                    },
                    {
                        "role": "user",
                        "content": question,
                    },
                ],
                "reward_model": {"style": "rule", "ground_truth": answer_raw},
                "extra_info": {
                    "split": "test",
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                    "need_tools_kwargs": True,
                    "tools_kwargs": tools_kwargs,
                    
                },
            }
            return data

        return process_fn

    math_data = math_data.map(function=make_map_fn(), with_indices=True)
    search_data = search_data.map(function=make_map_fn(), with_indices=True)

    eval_data = concatenate_datasets([math_data, search_data])

    eval_data.to_parquet('/workspace/verl/data/eval/eval_data.parquet')
