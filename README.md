# patentpick
a simple mailing service to deliver summary of patents from uspto
## how to start
### prepare config
```
# src/config.toml
[server]
uspto_url = "https://bulkdata.uspto.gov/data/patent/application/redbook/fulltext/"
uspto_pdf_url = "https://ppubs.uspto.gov/dirsearch-public/print/downloadPdf/"
uspto_year = "2024"

[localpath]
resources = "resources"
documents = "resources/documents"
checkpoints = "resources/AI-Growth-Lab_PatentSBERTa"

[vectordb]
qdrant_url = "http://localhost:6334"
vector_dim = 768
collection_name = "patents"
upload_chunk_size = 1000

[log]
level = "info"
```
### download model from huggingface
`git clone https://huggingface.co/AI-Growth-Lab/PatentSBERTa`
> This is a sentence-transformers model: It maps sentences & paragraphs to a 768 dimensional dense vector space and can be used for tasks like clustering or semantic search
- once the model downloaded, one can build it from `.safetensors` file.  
- [huggingface model repo](https://huggingface.co/AI-Growth-Lab/PatentSBERTa)

### convert pytorch to .safetensors

<details>

<summary>convert to safetensors</summary>
<br>

```
import json
import os
import sys
from collections import defaultdict
from tqdm import tqdm
import argparse
import torch

from safetensors.torch import load_file, save_file

def shared_pointers(tensors):
ptrs = defaultdict(list)
for k, v in tensors.items():
ptrs[v.data_ptr()].append(k)
failing = []
for ptr, names in ptrs.items():
if len(names) > 1:
failing.append(names)
return failing

def check_file_size(sf_filename: str, pt_filename: str):
sf_size = os.stat(sf_filename).st_size
pt_size = os.stat(pt_filename).st_size

    if (sf_size - pt_size) / pt_size > 0.01:
        raise RuntimeError(
            f"""The file size different is more than 1%:
         - {sf_filename}: {sf_size}
         - {pt_filename}: {pt_size}
         """
        )

def convert_file(
pt_filename: str,
sf_filename: str,
):
loaded = torch.load(pt_filename, map_location="cpu")
if "state_dict" in loaded:
loaded = loaded["state_dict"]
shared = shared_pointers(loaded)
for shared_weights in shared:
for name in shared_weights[1:]:
loaded.pop(name)

    # For tensors to be contiguous
    loaded = {k: v.contiguous().half() for k, v in loaded.items()}

    dirname = os.path.dirname(sf_filename)
    os.makedirs(dirname, exist_ok=True)
    save_file(loaded, sf_filename, metadata={"format": "pt"})
    check_file_size(sf_filename, pt_filename)
    reloaded = load_file(sf_filename)
    for k in loaded:
        pt_tensor = loaded[k]
        sf_tensor = reloaded[k]
        if not torch.equal(pt_tensor, sf_tensor):
            raise RuntimeError(f"The output tensors do not match for key {k}")

def rename(pt_filename: str) -> str:
filename, ext = os.path.splitext(pt_filename)
local = f"{filename}.safetensors"
local = local.replace("pytorch_model", "model")
return local

def convert_multi(folder: str, delprv: bool):
filename = "pytorch_model.bin.index.json"
with open(os.path.join(folder, filename), "r") as f:
data = json.load(f)

    filenames = set(data["weight_map"].values())
    local_filenames = []
    for filename in tqdm(filenames):
        pt_filename = os.path.join(folder, filename)
        sf_filename = rename(pt_filename)
        sf_filename = os.path.join(folder, sf_filename)
        convert_file(pt_filename, sf_filename)
        local_filenames.append(sf_filename)
        if(delprv):
            os.remove(pt_filename)

    index = os.path.join(folder, "model.safetensors.index.json")
    with open(index, "w") as f:
        newdata = {k: v for k, v in data.items()}
        newmap = {k: rename(v) for k, v in data["weight_map"].items()}
        newdata["weight_map"] = newmap
        json.dump(newdata, f, indent=4)
    local_filenames.append(index)
    if(delprv):
        os.remove(os.path.join(folder,"pytorch_model.bin.index.json"))
    return


def convert_single(folder: str, delprv: bool):
pt_name = "pytorch_model.bin"
pt_filename = os.path.join(folder, pt_name)
sf_name = "model.safetensors"
sf_filename = os.path.join(folder, sf_name)
convert_file(pt_filename, sf_filename)
if(delprv):
os.remove(pt_filename)
return

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', required=True, type=str, help="Path to the model dir")
parser.add_argument('-d', '--delete', default=False, required=False, type=bool, help="Delete pytorch files after conversion")
args = parser.parse_args()

for filename in os.listdir(args.model):
if filename == "pytorch_model.bin":
convert_single(args.model, args.delete)
sys.exit(0)
convert_multi(args.model, args.delete)
```

</details>

[take a look at this gist](https://gist.github.com/epicfilemcnulty/1f55fd96b08f8d4d6693293e37b4c55e#file-2safetensors-py)

## exmample
<img width="911" alt="image" src="https://github.com/NewBornRustacean/patentpick/assets/126950833/0631b5d8-90d3-4da4-b379-45e8cf140819">

## References
- [original embedding model: AI-Growth-Lab/PatentSBERTa](https://huggingface.co/AI-Growth-Lab/PatentSBERTa)
- [mpnet-rs](https://crates.io/crates/mpnet-rs)
- [Qdrant](https://qdrant.tech/documentation/)
