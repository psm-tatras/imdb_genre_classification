from utils import *
from model import *

# load all x_train and y_train data
x_train,y_train = load_partial_data(train_csv_file,n=1000)
print("Total %d x_train, %d y_train"%(len(x_train),len(y_train)))

# load label maps
label2ind = load_dict_from_json(label2ind_json)
ind2label = load_dict_from_json(ind2label_json)
print("Labels loaded")

nc = len(label2ind.keys())
print("Total %d classes"%nc)

# make a model
model = AttentionClassifier(nc)
print("Model created")
model.load_model("model/")
model.fit(x_train,y_train,label2ind)
