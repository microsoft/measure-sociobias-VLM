## A Unified Framework and Dataset for Assessing Societal Bias in Vision-Language Models (EMNLP Findings 2024)

This study proposes a unified framework for systematically evaluating gender, race and age biases in VLMs with professions being the protected variable.

### Code

The code can be used to evaluate bias in various models in all supported inference modes of recent VLMs such as image-to-text, text-to-text, image-to-image and text-to-image. Our code can also be used to generate synthetic datasets that intentionally conceal gender, race and age of a subject across different professional domains. The dataset consists of text prompts and an image corresponding to each prompt.

### Dataset

The dataset includes action-based descriptions of each profession and is used to evaluate various societal biases exhibited by vision-language models.

### Findings

Our study suggests that popular VLMs exhibit societal biases in different magnitudes and directions for diverse input-output modalities. Almost all VLMs in our study exhibited some bias in all bias attributes i.e. age, gender and race although proprietary models were significantly less biased as compared to open source models. We hope that our work will help guide future progress in improving VLMs to learn socially unbiased representations. 

### Instructions

* Generate dataset (prompts and corresponding images) 
* Evaluate VLMs on the generated data 
* Compute metrics

### Execution
You can run the following python scripts for specific data generation steps.

- Generate prompts for all professions using `generate_prompts.py`
- You can then use `better_filter_prompts.py` to select a prompt per professions
- Images can now be generated using `gen_images.py`
- Finally all directions can be evaluated with `startotxt_infer_with_cache.py` and `startoimg_infer_with_cache.py`
- Once this is all done, you can run additional cleanup and bias attribute id for generated images using `postprocess.py`
- Finally a large excel can be dumped using `process_json.py`

### Citation

You can cite our work as:

```
@inproceedings{sathe-etal-2024-unified,
    title = "A Unified Framework and Dataset for Assessing Societal Bias in Vision-Language Models",
    author = "Sathe, Ashutosh  and
      Jain, Prachi  and
      Sitaram, Sunayana",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.66/",
    doi = "10.18653/v1/2024.findings-emnlp.66",
    pages = "1208--1249"
}
```

### Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [`opencode@microsoft.com`](mailto:opencode@microsoft.com) with any additional questions or comments.


### Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.


### Privacy

You can read more about Microsoft's privacy statement [here](https://go.microsoft.com/fwlink/?LinkId=521839).

