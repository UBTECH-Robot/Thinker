<div align="center">
  <h1>Thinker: A vision-language foundation model for embodied intelligence</h1>

  <p>
    <a href="https://github.com/UBTECH-Robot/Thinker">
      <img src="https://img.shields.io/badge/Code-GitHub-blue?logo=github" alt="Released Code">
    </a>
    <a href="https://huggingface.co/UBTECH-Robotics/Thinker-4B">
      <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Checkpoints-HuggingFace-orange" alt="Checkpoints">
    </a>
    <a href="https://arxiv.org/abs/2601.21199">
      <img src="https://img.shields.io/badge/Paper-arXiv-red" alt="Tech Report">
    </a>
  </p>

  <img src="./assets/logo1.jpg" width="600" style="margin-top: 10px;"/>
</div>


## 🌟 Overview

We are pleased to open-source **Thinker**, a state-of-the-art vision-language foundation model specifically engineered for embodied intelligence.
While conventional VLMs often struggle with perspective confusion and temporal oversight, Thinker is designed to bridge the gap between general scene understanding and robust robot-centric task-level capabilities.
By leveraging high-quality dataset curation, multi-stage training, and reinforcement learning, Thinker exhibits advanced capabilities across four core dimensions:
**Task Planning** with future-state prediction, **Spatial Intelligence** grounded in an egocentric coordinate system, **Temporal Understanding** through historical state integration, and precise **Visual Grounding**.
Leveraging these capabilities, Thinker sets new records across 7 embodied AI benchmarks in Task Planning, Visual Grounding and Spatial Understanding, and significantly outperforms existing open-source, closed-source, and specialized baselines, showing its potential as a foundation for embodied intelligence and autonomous robotic decision-making.

<!--Thinker exhibits advanced capabilities in spatial perception, long-horizon video comprehension, and temporal reasoning.-->

<!--
<div align="center" style='display: flex; flex-direction: row'>
<img src="./assets/train_data.jpg" style="width: 31%;"/>
<img src="./assets/demo.gif" style="width: 61%"/>
</div>
-->

<div align="center" style="display: flex; justify-content: center; align-items: center; width: 100%; height: 100vh;">
    <img src="./assets/train_data.jpg" style="width: 31%; margin-right: 2%;" />
    <img src="./assets/demo.gif" style="width: 61%;" />
</div>



## Update

- **`2026-01-28`**: 🤗 [Thinker-4B](https://huggingface.co/UBTECH-Robotics/Thinker-4B) model checkpoint has been released in Huggingface.


## Quick Start

Clone this repo, and set up the environment with a few commands.

```bash
# The Thinker model requires transformers >= 4.57.0
pip install "transformers>=4.57.0"
```
<!--
```bash
git clone https://github.com/UBTECH-Robot/Thinker.git
cd Thinker

conda create -n Thinker python=3.11
conda activate Thinker
pip install -r requirements.txt
-->

The following contains a code snippet illustrating how to use our Thinker. More details can refer to inference.py.

```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


model = Qwen3VLForConditionalGeneration.from_pretrained(
    "UBTECH-Robotics/Thinker-4B", dtype="auto", device_map="auto"
)


processor = AutoProcessor.from_pretrained("UBTECH-Robotics/Thinker-4B")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "http://images.cocodataset.org/val2017/000000039769.jpg",
            },
            {"type": "text", "text": "Please point out the free space between two cats."},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```


## 🤗 Models

| Model Name           | Checkpoint                                                                         | Description                                |
| :------------------- | :--------------------------------------------------------------------------------- | :----------------------------------------- |
| **Thinker 4B** | [🤗 UBTECH-Robotics/Thinker-4B](https://huggingface.co/UBTECH-Robotics/Thinker-4B) | 4B parameter Instruct version of Thinker   |
| **Thinker Thinking 4B** | ⌛ *Coming soon* | 4B parameter Reasoning (Thinking) version  |

## Evaluation

More evaluation results and scripts will be added soon.

<div align="center">
<img src="./assets/thinker_performan_2.jpg" width="800"/>
<p>The performance comparison of Thinker-4B with models below 10B.</p>
</div>


<div align="center">
<img src="./assets/thinker_performan_1.jpg" width="800"/>
<p>The performance comparison of Thinker-4B with models above 10B.</p>
</div>



## License

This project is released under the [Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en).


## Citation 

If you find our paper and code useful in your research, please consider giving a star ⭐ and citation 📝 :)
```BibTeX
@article{UBTECH_Thinker_short_reprort,
      title={Thinker: A vision-language foundation model for embodied intelligence}, 
      author={Baiyu Pan and Daqin Luo and Junpeng Yang and Jiyuan Wang and Yixuan Zhang and Hailin Shi and Jichao Jiao},
	    journal={https://arxiv.org/abs/2601.21199},
      year={2025}
}
```
