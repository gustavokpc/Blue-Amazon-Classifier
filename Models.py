import pandas as pd
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, matthews_corrcoef
from huggingface_hub import notebook_login, login

from transformers import AutoTokenizer
from transformers import create_optimizer
from transformers import DataCollatorWithPadding
from transformers import TFAutoModelForSequenceClassification
from transformers.keras_callbacks import PushToHubCallback
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from keras import backend as K

# Login HuggingFace

login("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")

# Hyperparameters

batch_size = 16
num_epochs = 10
learning_rate = 1e-6

# models pt_br: "neuralmind/bert-base-portuguese-cased", "neuralmind/bert-large-portuguese-c"ased
# models en: "bert-base-uncased", "bert-large-uncased", "roberta-base", "roberta-large"

model_name = "neuralmind/bert-base-portuguese-cased"
dataset_language = 0      # 0 = pt_br ; 1 = en

# ------------------------------ FUNCIONS ------------------------------

if(dataset_language == 0):
    output_name = "pt_br_MODEL_" + model_name + "_LRATE_" + str(learning_rate) + "_EPOCHS_" + str(num_epochs)
else:
    output_name = "en_MODEL_" + model_name + "_LRATE_" + str(learning_rate) + "_EPOCHS_" + str(num_epochs)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# PT_BR
dataframe_completo_pt_br = pd.read_csv("Datasets/dataset_pt_br.csv")
ds_completo_pt_br = Dataset.from_pandas(dataframe_completo_pt_br)

#EN
dataframe_completo_en = pd.read_csv("Datasets/dataset_en.csv")
ds_completo_en = Dataset.from_pandas(dataframe_completo_en)

# PT_BR
y_pt_br = dataframe_completo_pt_br.loc[:,"label"]
X_pt_br = dataframe_completo_pt_br.loc[:,"question"]

X_pt_br_train, X_pt_br_to_split, y_pt_br_train, y_pt_br_to_split = train_test_split(X_pt_br, y_pt_br, test_size=0.2, random_state=7, stratify=y_pt_br)
X_pt_br_validation, X_pt_br_test, y_pt_br_validation, y_pt_br_test = train_test_split(X_pt_br_to_split, y_pt_br_to_split, test_size=0.5, random_state=7, stratify=y_pt_br_to_split)

train_pt_br_df = pd.concat([X_pt_br_train, y_pt_br_train], axis=1)
validation_pt_br_df = pd.concat([X_pt_br_validation, y_pt_br_validation], axis=1)
test_pt_br_df = pd.concat([X_pt_br_test, y_pt_br_test], axis=1)

#EN
y_en = dataframe_completo_en.loc[:,"label"]
X_en = dataframe_completo_en.loc[:,"question"]

X_en_train, X_en_to_split, y_en_train, y_en_to_split = train_test_split(X_en, y_en, test_size=0.2, random_state=7, stratify=y_en)
X_en_validation, X_en_test, y_en_validation, y_en_test = train_test_split(X_en_to_split, y_en_to_split, test_size=0.5, random_state=7, stratify=y_en_to_split)

train_en_df = pd.concat([X_en_train, y_en_train], axis=1)
validation_en_df = pd.concat([X_en_validation, y_en_validation], axis=1)
test_en_df = pd.concat([X_en_test, y_en_test], axis=1)

# PT_BR
train_pt_br_ds = Dataset.from_pandas(train_pt_br_df)
validation_pt_br_ds = Dataset.from_pandas(validation_pt_br_df)
test_pt_br_ds = Dataset.from_pandas(test_pt_br_df)

#EN
train_en_ds = Dataset.from_pandas(train_en_df)
validation_en_ds = Dataset.from_pandas(validation_en_df)
test_en_ds = Dataset.from_pandas(test_en_df)

def preprocess_function(examples):
    return tokenizer(examples["question"], truncation=True)

tokenized_pt_br_train_ds = train_pt_br_ds.map(preprocess_function, batched=True)
tokenized_pt_br_validation_ds = validation_pt_br_ds.map(preprocess_function, batched=True)
tokenized_pt_br_test_ds = test_pt_br_ds.map(preprocess_function, batched=True)

tokenized_en_train_ds = train_en_ds.map(preprocess_function, batched=True)
tokenized_en_validation_ds = validation_en_ds.map(preprocess_function, batched=True)
tokenized_en_test_ds = test_en_ds.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

if (dataset_language == 0):
    dataframe_completo = dataframe_completo_pt_br
    ds_completo = ds_completo_pt_br
    y = y_pt_br
    y_train = y_pt_br_train
    y_to_split = y_pt_br_to_split
    y_validation = y_pt_br_validation
    y_test = y_pt_br_test
    X = X_pt_br
    X_train = X_pt_br_train
    X_to_split = X_pt_br_to_split
    X_validation = X_pt_br_validation
    X_test = X_pt_br_test
    train_df = train_pt_br_df
    validation_df = validation_pt_br_df
    test_df = test_pt_br_df
    train_ds = train_pt_br_ds
    validation_ds = validation_pt_br_ds
    test_ds = test_pt_br_ds
    tokenized_train_ds = tokenized_pt_br_train_ds
    tokenized_validation_ds = tokenized_pt_br_validation_ds
    tokenized_test_ds = tokenized_pt_br_test_ds
else:
    dataframe_completo = dataframe_completo_en
    ds_completo = ds_completo_en
    y = y_en
    y_train = y_en_train
    y_to_split = y_en_to_split
    y_validation = y_en_validation
    y_test = y_en_test
    X = X_en
    X_train = X_en_train
    X_to_split = X_en_to_split
    X_validation = X_en_validation
    X_test = X_en_test
    train_df = train_en_df
    validation_df = validation_en_df
    test_df = test_en_df
    train_ds = train_en_ds
    validation_ds = validation_en_ds
    test_ds = test_en_ds
    tokenized_train_ds = tokenized_en_train_ds
    tokenized_validation_ds = tokenized_en_validation_ds
    tokenized_test_ds = tokenized_en_test_ds

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

# ------------------------------ FUNCIONS ------------------------------

batches_per_epoch = len(X_train) // batch_size
total_train_steps = int(batches_per_epoch * num_epochs)
optimizer, schedule = create_optimizer(init_lr=learning_rate, num_warmup_steps=0, num_train_steps=total_train_steps)

with tf.device('/CPU:0'):
    model = TFAutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, id2label=id2label, label2id=label2id
    )

    model.summary()

    tf_train_set = model.prepare_tf_dataset(
        tokenized_train_ds,
        shuffle=True,
        batch_size=16,
        collate_fn=data_collator,
    )

    tf_validation_set = model.prepare_tf_dataset(
        tokenized_validation_ds,
        shuffle=False,
        batch_size=16,
        collate_fn=data_collator,
    )

    model.compile(optimizer=optimizer, metrics=['accuracy', f1_m, precision_m, recall_m])  # No loss argument!

    push_to_hub_callback = PushToHubCallback(
        output_dir=output_name,
        tokenizer=tokenizer,
    )
    
    EarlyStoppingCheckpoint = EarlyStopping(monitor = "val_loss",
                                            patience = 2,
                                            verbose = 1,
                                            mode = "min"
                                            )

    callbacks = [EarlyStoppingCheckpoint, push_to_hub_callback]

with tf.device('/CPU:0'):
    model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=num_epochs, callbacks=callbacks)

val_loss = model.history.history['val_loss']
val_acc = model.history.history['val_accuracy']
val_f1_m = model.history.history['val_f1_m']
print(val_loss)
print(val_acc)
print(val_f1_m)

token_val = [str(i) for i in pd.DataFrame(X_test)['question'].values]
y_pred = []

# Choose a specific model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained("gustavokpc/bert-base-portuguese-cased_LRATE_8e-06_EPOCHS_6")   
# model = TFAutoModelForSequenceClassification.from_pretrained("gustavokpc/bert-base-portuguese-cased_LRATE_8e-06_EPOCHS_6")

for i in range(len(token_val)):
    inputs = tokenizer(token_val[i], return_tensors="tf")
    logits = model(**inputs).logits
    prediction = int(tf.math.argmax(logits, axis=-1)[0])
    y_pred.append(prediction)

# plot the confusion matrix
mat = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 7))
sns.heatmap(mat.T, square=True, annot=True, fmt="d")
plt.xlabel("true labels")
plt.ylabel("predicted label")
plt.show()
print("The accuracy is {}".format(accuracy_score(y_test, y_pred)))
print("The matthews_corrcoef is {}".format(matthews_corrcoef(y_test, y_pred)))
print("The f1_score is {}".format(f1_score(y_test, y_pred, average='weighted')))
