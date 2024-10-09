import os
from collections import defaultdict
from typing import List, Dict
import lang2vec.lang2vec as l2v
import numpy as np
import torch
from torch import nn


class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)


def freeze_only_big_weights(model: nn.Module):
    """ post-processing on the model,
    freeze the model - train adapters later,
    cast the small parameters (e.g. layernorm) to fp32 for stability
    """
    for param in model.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)


def get_typological_language_features(lang_list: List[str],
                                      language_features: str = "syntax_knn+phonology_knn+inventory_knn",
                                      ) -> Dict[str, np.ndarray]:
    features = dict()
    for lang in lang_list:
        feat = l2v.get_features(l2v.LETTER_CODES[lang], language_features)[l2v.LETTER_CODES[lang]]
        features[lang] = np.asarray(feat)
    return features


def load_fasttext_model():
    import fasttext

    # model should be available at the dir before loading...
    ft_model_dir = os.environ.get("FASTTEXT_MODEL_DIR")
    global_lang = os.environ.get('GLOBAL_LANG_CODE')
    ft_model_name = "cc.%s.300.bin" % global_lang
    ft_model_path = os.path.join(ft_model_dir, ft_model_name)
    ft = fasttext.load_model(ft_model_path)
    return ft


class ResultDisplayer:
    def __init__(self, raw_result: str):
        self.raw_result = raw_result
        lines = self.raw_result.split('\n')
        parsed_results = defaultdict(list)
        self.lang_order = {
            'xnli': ['ar', 'hi', 'es', 'ru', 'sw', 'tr', 'bg', 'de', 'el', 'en', 'fr', 'th', 'ur', 'vi', 'zh'],
            'xcopa': ['tr', 'sw', 'et', 'id', 'it', 'ta', 'th', 'vi', 'ht', 'qu', 'zh'],
            'xstorycloze': ['ar', 'es', 'hi', 'ru', 'sw', 'en', 'eu', 'id', 'my', 'te', 'zh'],
            'xwinograd': ['jp', 'ru', 'en', 'fr', 'pt', 'zh'],
            'paws': ['es', 'ja', 'ko', 'de', 'en', 'fr', 'zh']
        }

        self.tasks = ['xnli',
                      'xcopa',
                      'xstorycloze',
                      'xwinograd',
                      'paws', ]

        # Iterate through each line
        for line in lines:
            # Split the line into columns
            columns = line.split('|')

            task_name = columns[1].strip()
            task_components = task_name.split('_')
            if len(task_components) != 2:
                continue

            task_id = task_components[0]
            task_id = task_id.split('-')[-1].strip()

            if task_id not in self.tasks:
                continue

            task_lang = task_components[1]

            value, std_err = float(columns[-4].strip()), float(columns[-2].strip())

            # Store the extracted information
            parsed_results[task_id].append({
                'task_lang': task_lang,
                'value': value,
                'std_err': std_err
            })

        self.parsed_results = parsed_results

    def display_results(self):
        for task_id, val_list in self.parsed_results.items():
            lang_order = self.lang_order[task_id]
            row_str = ''
            for lang in lang_order:
                current_val = None
                for val in val_list:
                    if val['task_lang'] == lang:
                        current_val = val
                        break

                acc = current_val['value']
                std_err = current_val['std_err']
                row_str += f'{acc}±{std_err},'
            print('------------')
            print(task_id)
            print(row_str)


if __name__ == '__main__':
    # lang_list = ['ar', 'ja', 'ru', 'hi', 'sw', 'es', 'tr']
    # features = get_typological_language_features(lang_list)
    # print(features)
    sample_result = """|pawsx            |N/A    |none  |     0|acc   |0.5223|±  |0.0233|
| - paws_de       |      0|none  |     0|acc   |0.5305|±  |0.0112|
| - paws_en       |      0|none  |     0|acc   |0.5220|±  |0.0112|
| - paws_es       |      0|none  |     0|acc   |0.4540|±  |0.0111|
| - paws_fr       |      0|none  |     0|acc   |0.5440|±  |0.0111|
| - paws_ja       |      0|none  |     0|acc   |0.5305|±  |0.0112|
| - paws_ko       |      0|none  |     0|acc   |0.5405|±  |0.0111|
| - paws_zh       |      0|none  |     0|acc   |0.5345|±  |0.0112|
|xcopa            |N/A    |none  |     0|acc   |0.5382|±  |0.0277|
| - xcopa_et      |      1|none  |     0|acc   |0.5180|±  |0.0224|
| - xcopa_ht      |      1|none  |     0|acc   |0.5320|±  |0.0223|
| - xcopa_id      |      1|none  |     0|acc   |0.5200|±  |0.0224|
| - xcopa_it      |      1|none  |     0|acc   |0.5580|±  |0.0222|
| - xcopa_qu      |      1|none  |     0|acc   |0.5140|±  |0.0224|
| - xcopa_sw      |      1|none  |     0|acc   |0.5440|±  |0.0223|
| - xcopa_ta      |      1|none  |     0|acc   |0.5500|±  |0.0223|
| - xcopa_th      |      1|none  |     0|acc   |0.5660|±  |0.0222|
| - xcopa_tr      |      1|none  |     0|acc   |0.5640|±  |0.0222|
| - xcopa_vi      |      1|none  |     0|acc   |0.5100|±  |0.0224|
| - xcopa_zh      |      1|none  |     0|acc   |0.5440|±  |0.0223|
|xnli             |N/A    |none  |     0|acc   |0.3605|±  |0.0248|
| - xnli_ar       |      1|none  |     0|acc   |0.3402|±  |0.0095|
| - xnli_bg       |      1|none  |     0|acc   |0.3876|±  |0.0098|
| - xnli_de       |      1|none  |     0|acc   |0.3928|±  |0.0098|
| - xnli_el       |      1|none  |     0|acc   |0.3305|±  |0.0094|
| - xnli_en       |      1|none  |     0|acc   |0.4373|±  |0.0099|
| - xnli_es       |      1|none  |     0|acc   |0.3506|±  |0.0096|
| - xnli_fr       |      1|none  |     0|acc   |0.3647|±  |0.0096|
| - xnli_hi       |      1|none  |     0|acc   |0.3639|±  |0.0096|
| - xnli_ru       |      1|none  |     0|acc   |0.3835|±  |0.0097|
| - xnli_sw       |      1|none  |     0|acc   |0.3526|±  |0.0096|
| - xnli_th       |      1|none  |     0|acc   |0.3490|±  |0.0096|
| - xnli_tr       |      1|none  |     0|acc   |0.3349|±  |0.0095|
| - xnli_ur       |      1|none  |     0|acc   |0.3470|±  |0.0095|
| - xnli_vi       |      1|none  |     0|acc   |0.3382|±  |0.0095|
| - xnli_zh       |      1|none  |     0|acc   |0.3345|±  |0.0095|
|xstorycloze      |N/A    |none  |     0|acc   |0.5263|±  |0.0199|
| - xstorycloze_ar|      1|none  |     0|acc   |0.5069|±  |0.0129|
| - xstorycloze_en|      1|none  |     0|acc   |0.5539|±  |0.0128|
| - xstorycloze_es|      1|none  |     0|acc   |0.5387|±  |0.0128|
| - xstorycloze_eu|      1|none  |     0|acc   |0.5175|±  |0.0129|
| - xstorycloze_hi|      1|none  |     0|acc   |0.5189|±  |0.0129|
| - xstorycloze_id|      1|none  |     0|acc   |0.5334|±  |0.0128|
| - xstorycloze_my|      1|none  |     0|acc   |0.5030|±  |0.0129|
| - xstorycloze_ru|      1|none  |     0|acc   |0.5420|±  |0.0128|
| - xstorycloze_sw|      1|none  |     0|acc   |0.5347|±  |0.0128|
| - xstorycloze_te|      1|none  |     0|acc   |0.5447|±  |0.0128|
| - xstorycloze_zh|      1|none  |     0|acc   |0.4950|±  |0.0129|
|xwinograd        |N/A    |none  |     0|acc   |0.5096|±  |0.0212|
| - xwinograd_en  |      1|none  |     0|acc   |0.5071|±  |0.0104|
| - xwinograd_fr  |      1|none  |     0|acc   |0.5422|±  |0.0550|
| - xwinograd_jp  |      1|none  |     0|acc   |0.5089|±  |0.0162|
| - xwinograd_pt  |      1|none  |     0|acc   |0.4791|±  |0.0309|
| - xwinograd_ru  |      1|none  |     0|acc   |0.5111|±  |0.0282|
| - xwinograd_zh  |      1|none  |     0|acc   |0.5317|±  |0.0222|"""
    displayer = ResultDisplayer(sample_result)
    displayer.parsed_results
    displayer.display_results()
