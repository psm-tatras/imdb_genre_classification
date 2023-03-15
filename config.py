hub_preprocessor_model = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
hub_transformer_model = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/2"
train_csv_file = "data/imdb_genre_train.csv"
label2ind_json = "data/label2ind.json"
ind2label_json = "data/ind2label.json"
attention_dim = 128 # should come from transformer dim