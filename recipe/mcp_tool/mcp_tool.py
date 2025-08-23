from verl.utils.reward_score import mcp_tools1

def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict) -> float:

    # encourage model to call tools
    # num_turns = extra_info["num_turns"]
    # if result["score"] < 0:
    #     tool_call_reward = (num_turns - 2) / 2 * 0.1
    #     result["score"] = min(0, result["score"] + tool_call_reward)

    # if result["pred"] is None:
    #     result["pred"] = ""

    # 可以添加其他逻辑

    # We change predict_str to solution_str to match the caller,
    # and pass it to the predict_str argument of the downstream function.
    return mcp_tools1.compute_score(data_source, solution_str, ground_truth, use_boxed=True)
