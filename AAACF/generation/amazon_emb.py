import json
import openai
import numpy as np
from tqdm import tqdm
import pickle  # 导入pickle模块


client = openai.OpenAI(
    api_key="",
)

def get_gpt_emb(prompt):

    embedding = client.embeddings.create(
        input=prompt,
        model="text-embedding-ada-002"
    ).data[0].embedding

    return np.array(embedding)

# Read generated profiles
profiles = []

with open('../data/amazon-CD/profile_user.jsonl', 'r') as f:
    for line in f.readlines():
        profiles.append(json.loads(line))

# List to store embeddings
embeddings = []

# Iterate over all profiles and encode them
for profile in tqdm(profiles, desc="Encoding Profiles", unit="profile"):
    emb = get_gpt_emb(profile['profile'])
    embeddings.append(emb)

# Convert list of embeddings to a NumPy array
embeddings_array = np.array(embeddings)

# Save embeddings to a .pkl file
with open('../data/amazon-CD/usr_emb_np.pkl', 'wb') as f:
    pickle.dump(embeddings_array, f)

print("Embeddings saved to 'embeddings.pkl'")
#
# class Colors:
#     GREEN = '\033[92m'
#     END = '\033[0m'
#
# print(Colors.GREEN + "Encoding Semantic Representation" + Colors.END)
# print("---------------------------------------------------\n")
# print(Colors.GREEN + "The Profile is:\n" + Colors.END)
# print(profiles[0]['profile'])
# print("---------------------------------------------------\n")
# emb = get_gpt_emb(profiles[0]['profile'])
# print(Colors.GREEN + "Encoded Semantic Representation Shape:" + Colors.END)
# print(emb.shape)
# print(emb)
