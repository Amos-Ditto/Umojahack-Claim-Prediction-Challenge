import pandas as pd
# import numpy as np
# from tensorflow import keras
import tensorflow as tf


path_train = 'data/train.csv'

df = pd.read_csv(path_train)

df = df.drop(['policy_number'], axis=1)
df = df.drop(['loss_date'], axis=1)
df = df.drop(['claim_id'], axis=1)
df = df.drop(['claim_number'], axis=1)



maxlen = df['target'].unique().max()
print(maxlen)

def dataframe_to_ds(dataframe):
    data = dataframe.copy()
    labels = data.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(data), labels))
    ds = ds.batch(64)

    return ds

train_dataset = df[:7000]
val_dataset = df[7000:]

dataset = dataframe_to_ds(train_dataset)
val_dataset = dataframe_to_ds(val_dataset)

print(dataset.take(1).shape)
# # for x, y in dataset.take(1):
# #     print(x)
# #     print('\n')
# #     print(y)


# def encode_numerical_feature(feature, name, dataset):
#     normalizer = keras.layers.Normalization()

#     feature_ds = dataset.map(lambda x, y: x[name])
#     feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

#     normalizer.adapt(feature_ds)

#     encoded_feature = normalizer(feature)
#     return encoded_feature

# def encode_categorical_feature(feature, name, dataset, is_string):
#     lookup_class = keras.layers.StringLookup if is_string else keras.layers.IntegerLookup
#     lookup = lookup_class(output_mode="binary")

#     feature_ds = dataset.map(lambda x, y: x[name])
#     feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

#     lookup.adapt(feature_ds)

#     encoded_feature = lookup(feature)
#     return encoded_feature

# sum_insured = keras.Input(shape=(1,), name="sum_insured")
# product = keras.Input(shape=(1,), name="product", dtype="string")
# agent = keras.Input(shape=(1,), name="agent", dtype="string")
# class_of_business = keras.Input(shape=(1,), name="class_of_business", dtype="string")
# risk_type = keras.Input(shape=(1,), name="risk_type", dtype="string")
# client_type = keras.Input(shape=(1,), name="client_type", dtype="string")
# renewal_frequency = keras.Input(shape=(1,), name="renewal_frequency", dtype="string")
# primary_cause = keras.Input(shape=(1,), name="primary_cause", dtype="string")
# secondary_cause = keras.Input(shape=(1,), name="secondary_cause", dtype="string")
# branch = keras.Input(shape=(1,), name="branch", dtype="string")


# all_inputs = [
#     sum_insured,
#     product,
#     agent,
#     class_of_business,
#     risk_type,
#     client_type,
#     renewal_frequency,
#     primary_cause,
#     secondary_cause,
#     branch
# ]

# sum_insured_enc = encode_numerical_feature(sum_insured, 'sum_insured', dataset)
# product_enc = encode_categorical_feature(product, 'product', dataset, True)

# agent_enc = encode_categorical_feature(agent, 'agent', dataset, True)
# class_of_business_enc = encode_categorical_feature(class_of_business, 'class_of_business', dataset, True)
# risk_type_enc = encode_categorical_feature(risk_type, 'risk_type', dataset, True)
# client_type_enc = encode_categorical_feature(client_type, 'client_type', dataset, True)
# renewal_frequence_enc = encode_categorical_feature(renewal_frequency, 'renewal_frequency', dataset, True)
# primary_cause_enc = encode_categorical_feature(primary_cause, 'primary_cause', dataset, True)
# branch_enc = encode_categorical_feature(branch, 'branch', dataset, True)

# all_features = keras.layers.concatenate([
#     sum_insured_enc,
#     product_enc,
#     agent_enc,
#     class_of_business_enc,
#     risk_type_enc,
#     client_type_enc,
#     renewal_frequence_enc,
#     primary_cause_enc,
#     branch_enc
# ])

# x = keras.layers.Dense(20, activation="relu")(all_features)
# x = keras.layers.Dense(10, activation="relu")(x)
# # x = keras.layers.Flatten()
# x = keras.layers.Dense(6545, activation="linear")(x)

# model = keras.Model(all_inputs, x)

# model.compile(optimizer="adam", loss='binary_crossentropy', metrics=["accuracy"])

# model.fit(dataset, epochs=10, validation_data=val_dataset)


