import re

def extract_answer(predict_str: str) -> str:
    pattern = re.compile(r"\\boxed\{\{(.*?)\}\}", re.DOTALL)

    match = pattern.search(predict_str)
    if match:
        answer = match.group(1)
        print("提取出的答案:", answer)
    else:
        print("没有匹配到答案")


def compute_score(data_source: str, predict_str: str, ground_truth: str, use_boxed: bool = True) -> float:
    
    print("===========ROLLOUT_DEBUG=========")
    print(predict_str)
    print("===========ROLLOUT_DEBUG=========")

    answer = extract_answer(predict_str)
    # print(answer)
    
    return 1.0 if answer == ground_truth else 0.0
