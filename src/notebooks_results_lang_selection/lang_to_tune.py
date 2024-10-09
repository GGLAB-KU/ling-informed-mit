import dataclasses
import os
import pickle
import random
from collections import Counter
from pprint import pprint
from typing import Optional, List

import lang2vec.lang2vec as l2v
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin_min
from tqdm import tqdm
from dataclasses import dataclass

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


@dataclass
class LangToTune:
    code_two: str
    code_three: str
    typology: np.ndarray  # "syntax_knn+phonology_knn+inventory_knn"
    learned: Optional[np.ndarray]
    geo: np.ndarray
    family: str
    subfamily: Optional[str]
    rest_of_family: List[str]
    svo_orders: List[str]


class BacktrianLangSelector:
    def __init__(self,
                 lang_properties_csv: str = 'lang_properties.csv'):
        self.saved_file = 'backtrian_lang_selector.pickle'
        self.lang_properties_csv = lang_properties_csv
        if os.path.exists(self.saved_file):
            self.load()
        else:
            self.initialize()
            self.save()

    def initialize(self):
        langs_str \
            = 'af,ar,az,bn,cs,de,en,es,et,fa,fi,fr,gl,gu,he,hi,hr,id,it,ja,ka,kk,km,ko,lt,lv,mk,ml,mn,mr,my,ne,nl,pl,ps,pt,ro,ru,si,sl,sv,sw,ta,te,th,tl,tr,uk,ur,vi,xh,zh'
        langs = langs_str.split(',')
        letter_codes = list(map(lambda lang: (lang, l2v.LETTER_CODES[lang]), langs))

        self.processed_langs = []

        for code_two, letter_code in tqdm(letter_codes, desc='extracting lang features...'):
            # typology features
            feature_set = "syntax_knn+phonology_knn+inventory_knn"
            feat = l2v.get_features(letter_code, feature_set)[letter_code]
            synt_phono_inventory_features = np.asarray(feat)

            # learned features
            feature_set = "learned"
            if letter_code in l2v.LEARNED_LANGUAGES:
                learned_features = l2v.get_features(letter_code, feature_set)[letter_code]
                learned_features = np.asarray(learned_features)
            else:
                print(f'{letter_code} not in URIEL Languages')
                learned_features = None

            # geo
            features = l2v.get_features(letter_code, "geo", header=False)
            geo_features = features[letter_code]

            # family
            features = l2v.get_features(letter_code, "fam", header=True)
            families_and_subfamilies = np.array(features['CODE'])[np.array(features[letter_code]) == 1]
            family = families_and_subfamilies[0]
            subfamily = None
            rest_of_family = []
            if len(families_and_subfamilies) > 1:
                subfamily = families_and_subfamilies[1]
            if len(families_and_subfamilies) > 2:
                rest_of_family = families_and_subfamilies[2:]

            # svo orders
            features = l2v.get_features(letter_code, "syntax_knn", header=True)
            for k, v in features.items():
                features[k] = v[:5]
            feature_map = np.array(list(map(lambda x: x if x != '--' else 0, features[letter_code]))) == 1
            svo_orders = np.array(features['CODE'])[feature_map == 1]

            lang = LangToTune(code_two,
                              letter_code,
                              synt_phono_inventory_features,
                              learned_features,
                              geo_features,
                              family,
                              subfamily,
                              rest_of_family,
                              svo_orders)
            self.processed_langs.append(lang)

    def _cluster_with_kmeans(self, X, num_clusters: int, random_state=42):
        kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
        kmeans.fit(X)
        return kmeans

    def get_typology_selections(self, num_selections: int, random_state: int = 66):
        typology_features = list(map(lambda x: x.typology, self.processed_langs))
        typology_features = np.array(typology_features)
        kmeans = self._cluster_with_kmeans(typology_features, num_selections, random_state=random_state)
        centroids = kmeans.cluster_centers_
        selected_langs = []
        for centroid in centroids:
            # Find the closest point and its distance for each centroid
            index, distance = pairwise_distances_argmin_min(centroid.reshape(1, -1), typology_features)
            selected_langs.append(self.processed_langs[index[0]])
        return selected_langs

    def get_learned_selections(self, num_selections: int, random_state: int = 42):
        avail_langs = list(filter(lambda lang: lang.learned is not None, self.processed_langs))
        features = list(map(lambda x: x.learned, avail_langs))
        features = np.array(features)
        kmeans = self._cluster_with_kmeans(features, num_selections, random_state=random_state)
        centroids = kmeans.cluster_centers_
        selected_langs = []
        for centroid in centroids:
            # Find the closest point and its distance for each centroid
            index, distance = pairwise_distances_argmin_min(centroid.reshape(1, -1), features)
            selected_langs.append(avail_langs[index[0]])
        return selected_langs

    def get_geo_selections(self, num_selections: int, random_state: int = 42):
        geo = list(map(lambda x: x.geo, self.processed_langs))
        geo = np.array(geo)
        kmeans = self._cluster_with_kmeans(geo, num_selections, random_state=random_state)
        centroids = kmeans.cluster_centers_
        selected_langs = []
        for centroid in centroids:
            # Find the closest point and its distance for each centroid
            index, distance = pairwise_distances_argmin_min(centroid.reshape(1, -1), geo)
            selected_langs.append(self.processed_langs[index[0]])
        return selected_langs

    def get_lang_family_selections(self,
                                   num_selections: int = 14,
                                   random_state: int = 42):
        df = pd.read_csv(self.lang_properties_csv)
        df = df.dropna(subset=['Language'])
        # Shuffle the DataFrame to randomize the order of rows
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        # We use drop_duplicates to ensure one language per family
        selected_languages = df.drop_duplicates(subset='language family').head(num_selections)

        lang_code_twos = list(map(lambda x: x[1][0], selected_languages[['ISO 639-1\nCode']].iterrows()))

        result = list(filter(
            lambda x: x.code_two in lang_code_twos,
            self.processed_langs
        ))

        assert len(result) == num_selections, "Number of selected languages does not match"

        return sorted(list(map(lambda x: x.code_two, result)))

    def get_random_selections(self,
                              num_selections: int,
                              seed: int = 66):
        random.seed(seed)
        return random.sample(self.processed_langs, num_selections)

    def get_colex2lang_selections(self, num_selections: int, random_state: int = 42):
        embeddings = {}
        with open('clics_prone_embeddings', 'r') as file:
            next(file)
            for line in file:
                parts = line.strip().split()
                token = parts[0]
                vector = np.array(parts[1:], dtype=np.float32)
                embeddings[token] = vector

        embeddings_array = []
        langs_pool = []
        for lang in self.processed_langs:
            embedding = embeddings.get(lang.code_three)
            if embedding is not None:
                langs_pool.append(lang)
                embeddings_array.append(embedding)

        features = np.array(embeddings_array)
        kmeans = self._cluster_with_kmeans(features, num_selections, random_state=random_state)
        centroids = kmeans.cluster_centers_
        selected_langs = []
        for centroid in centroids:
            # Find the closest point and its distance for each centroid
            index, distance = pairwise_distances_argmin_min(centroid.reshape(1, -1), features)
            selected_langs.append(langs_pool[index[0]])
        return selected_langs

    def save(self):
        with open(self.saved_file, 'wb') as file:
            pickle.dump(self.processed_langs, file)

    def load(self):
        with open(self.saved_file, 'rb') as file:
            self.processed_langs = pickle.load(file)

    def visualize_geo_features(self):
        geo = list(map(lambda x: x.geo, self.processed_langs))
        geo = np.array(geo)

        processed_langs = self.processed_langs

        kmeans = self._cluster_with_kmeans(geo, 14)
        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_
        # labels = list(map(lambda x: x * 2, labels))

        # Find and mark the selected languages
        selected_langs = []
        # Apply t-SNE to reduce dimensions for visualization
        tsne = TSNE(n_components=2, perplexity=40, n_iter=3000, random_state=42)
        geo_tsne = tsne.fit_transform(geo)

        # Plot all the points using t-SNE 2D vectors
        plt.figure(figsize=(14, 10))
        scatter = plt.scatter(geo_tsne[:, 0], geo_tsne[:, 1], c=labels, cmap='tab20c', alpha=1)

        # Find the closest real data points to each centroid
        for i, centroid in enumerate(centroids):
            index, _ = pairwise_distances_argmin_min(centroid.reshape(1, -1), geo)
            selected_langs.append(processed_langs[index[0]])
            # plt.scatter(geo_tsne[index, 0], geo_tsne[index, 1], c='red', label='Selected' if i == 0 else "", marker='X',
            #             alpha=1)

        for i, lang in enumerate(processed_langs):
            if lang in selected_langs:
                plt.annotate(lang.code_two,
                             (geo_tsne[i, 0], geo_tsne[i, 1]),
                             color='red',
                             weight='black',
                             fontsize='large')
            else:
                plt.annotate(lang.code_two,
                             (geo_tsne[i, 0], geo_tsne[i, 1]),
                             weight='heavy',
                             fontsize='medium')

        # Add legend and labels
        # plt.legend()
        # plt.title('t-SNE Visualization of Geographical Feature Vectors')
        plt.xlabel('comp-1')
        plt.ylabel('comp-2')

        plt.savefig('geo_feat_tsne_clustering.png',
                    dpi=300,
                    bbox_inches='tight')

        plt.show()


def lang_selection_for_main_experiments():
    lang_selector = BacktrianLangSelector()
    random.seed(0)

    all_languages = set()

    random_family_selection = lang_selector.get_lang_family_selections(14, random_state=66)
    selections = [random_family_selection]
    pprint('**** RANDOM FAMILY SELECTIONS ***')
    [pprint(','.join(selection)) for selection in selections]
    [all_languages.update(selection) for selection in selections]

    seeds = [66, 42, 10]
    selections = []
    for i in seeds:
        typo_selected_langs = lang_selector.get_typology_selections(14, random_state=i)
        res = sorted(list(map(lambda x: x.code_two, typo_selected_langs)))
        selections.append(res)

    pprint('**** TYPOLOGY SELECTIONS ***')
    [pprint(','.join(selection)) for selection in selections]
    [all_languages.update(selection) for selection in selections]

    flattened_list = [item for sublist in selections for item in sublist]
    most_common_14 = Counter(flattened_list).most_common(14)
    most_common_14 = list(map(lambda x: x[0], sorted(most_common_14, key=lambda x: x[0])))
    print(most_common_14)

    selections = []
    for i in seeds:
        learned_selected_langs = lang_selector.get_learned_selections(14, random_state=i)
        res = sorted(list(map(lambda x: x.code_two, learned_selected_langs)))
        selections.append(res)

    pprint('**** LEARN SELECTIONS ***')
    [pprint(','.join(selection)) for selection in selections]
    [all_languages.update(selection) for selection in selections]

    flattened_list = [item for sublist in selections for item in sublist]
    most_common_14 = Counter(flattened_list).most_common(14)
    most_common_14 = list(map(lambda x: x[0], sorted(most_common_14, key=lambda x: x[0])))
    print(most_common_14)

    selections = []
    for i in seeds:
        geo_selected_langs = lang_selector.get_geo_selections(14, random_state=i)
        res = sorted(list(map(lambda x: x.code_two, geo_selected_langs)))
        selections.append(res)

    pprint('**** GEO SELECTIONS ***')
    [pprint(','.join(selection)) for selection in selections]
    [all_languages.update(selection) for selection in selections]

    flattened_list = [item for sublist in selections for item in sublist]
    most_common_14 = Counter(flattened_list).most_common(14)
    most_common_14 = list(map(lambda x: x[0], sorted(most_common_14, key=lambda x: x[0])))
    print(most_common_14)

    selections = []
    for i in seeds:
        colex2lang_selections = lang_selector.get_colex2lang_selections(14, random_state=i)
        res = sorted(list(map(lambda x: x.code_two, colex2lang_selections)))
        selections.append(res)

    pprint('**** SEMANTIC SELECTIONS ***')
    [pprint(','.join(selection)) for selection in selections]
    [all_languages.update(selection) for selection in selections]

    flattened_list = [item for sublist in selections for item in sublist]
    most_common_14 = Counter(flattened_list).most_common(14)
    most_common_14 = list(map(lambda x: x[0], sorted(most_common_14, key=lambda x: x[0])))
    print(most_common_14)

    selections = []
    for i in seeds:
        random_selections = lang_selector.get_random_selections(14, i)
        res = sorted(list(map(lambda x: x.code_two, random_selections)))
        selections.append(res)

    pprint('**** RANDOM SELECTIONS ***')
    [pprint(','.join(selection)) for selection in selections]
    [all_languages.update(selection) for selection in selections]

    print(all_languages)


def varying_number_langs_for_geo(output_file: str = 'geo_language_selections.txt'):
    lang_selector = BacktrianLangSelector()
    random.seed(0)

    seeds = [66, 42, 10]

    # in total we have 52 languages
    # ALL is already computed, 14 is already computed
    num_langs = list(range(1, 14)) + [20, 26]

    with open(output_file, 'w') as file:
        for num_lang in num_langs:
            file.write(f'**** LANGUAGE COUNT UPDATED, {num_lang} languages\n')
            selections = []
            for seed_num, i in enumerate(seeds):
                geo_selected_langs = lang_selector.get_geo_selections(num_lang, random_state=i)
                res = sorted(list(map(lambda x: x.code_two, geo_selected_langs)))
                selections.append(res)

                # file.write(f'seed id: {str(seed_num)} - seed: {str(i)} ***\n')
                file.write(','.join(res) + '\n')


if __name__ == '__main__':
    lang_selection_for_main_experiments()
    # varying_number_langs_for_geo()
