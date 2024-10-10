import json
import openai
import numpy as np
from tqdm import tqdm

client = openai.OpenAI(
    api_key="",
)


def get_gpt_response_w_system(prompt):
    global system_prompt
    completion = client.chat.completions.create(
        model='gpt-3.5-turbo-ca',
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    )
    response = completion.choices[0].message.content

    # 移除可能存在的三引号
    if response.startswith("```") and response.endswith("```"):
        response = response[3:-3].strip()

    return response

# read the system_prompt (Instruction) for user profile generation
system_prompt = ""
with open('../instruction/amazon_usr.txt', 'r') as f:
    for line in f.readlines():
        system_prompt += line

# read the example prompts of users
example_prompts = []
with open('../data/amazon-CD/prompts_user.json', 'r') as f:
    for line in f.readlines():
        u_prompt = json.loads(line)
        example_prompts.append(u_prompt['prompt'])


# 创建一个文件来存储所有用户的返回结果
with open('../data/amazon-CD/profile_user.jsonl', 'w') as output_file:
    for idx, prompt in tqdm(enumerate(example_prompts), desc="Processing users"):
        response = get_gpt_response_w_system(prompt)
        try:
            response_json = json.loads(response)

            user_response = {
                # "user_id": idx,
                "profile": response_json.get("summarization", ""),
                "reasoning": response_json.get("reasoning", "")
            }
            # print(user_response)
            # 将每个用户的响应结果写入文件，每行一个字典
        except Exception as e:
            print(f"Error processing prompt {idx}: {e}")
            print(response)
            user_response = {
                # "user_id": idx,
                "profile": "None",
                "reasoning": "None"
            }
        output_file.write(json.dumps(user_response) + "\n")

print("All user responses have been written to user_responses.jsonl")

# indexs = len(example_prompts)
# picked_id = np.random.choice(indexs, size=1)[0]
#
# class Colors:
#     GREEN = '\033[92m'
#     END = '\033[0m'
#
# print(Colors.GREEN + "Generating Profile for User" + Colors.END)
# print("---------------------------------------------------\n")
# print(Colors.GREEN + "The System Prompt (Instruction) is:\n" + Colors.END)
# print(system_prompt)
# print("---------------------------------------------------\n")
# print(Colors.GREEN + "The Input Prompt is:\n" + Colors.END)
# print(example_prompts[picked_id])
# print("---------------------------------------------------\n")
# response = get_gpt_response_w_system(example_prompts[picked_id])
# print(Colors.GREEN + "Generated Results:\n" + Colors.END)
# print(response)