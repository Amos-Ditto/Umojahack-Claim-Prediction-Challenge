import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np


path_train = 'data/train.csv'

df = pd.read_csv(path_train)

df = df.drop(['policy_number'], axis=1)
df = df.drop(['loss_date'], axis=1)
df = df.drop(['claim_id'], axis=1)
df = df.drop(['claim_number'], axis=1)

maxlen = len(df['target'].unique())
df['target'] = df['target'] / 7868590.62

def dataframe_to_ds(dataframe):
    data = dataframe.copy()
    labels = data.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(data), labels))
    ds = ds.batch(64)

    return ds

train_dataset = df[:7000]
val_dataset = df[7000:]

x_train = dataframe_to_ds(train_dataset)
val_dataset = dataframe_to_ds(val_dataset)



class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [keras.layers.Dense(ff_dim, activation="relu"), keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)



class TokenAndPositionEmbedding(keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        # self.pos_emb = keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        # maxlen = x.shape
        # positions = tf.range(start=0, limit=maxlen[-1], delta=1)
        # positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x #+ positions



def encode_numerical_feature(feature, name, dataset):
    normalizer = keras.layers.Normalization()

    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    normalizer.adapt(feature_ds)

    encoded_feature = normalizer(feature)
    return encoded_feature

def encode_categorical_feature(feature, name, dataset, is_string):
    lookup_class = keras.layers.StringLookup if is_string else keras.layers.IntegerLookup
    lookup = lookup_class(output_mode="binary")

    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    lookup.adapt(feature_ds)
    
    encoded_feature = lookup(feature)
    return encoded_feature

sum_insured = keras.Input(shape=(1,), name="sum_insured")
product = keras.Input(shape=(1,), name="product", dtype="string")
agent = keras.Input(shape=(1,), name="agent", dtype="string")
class_of_business = keras.Input(shape=(1,), name="class_of_business", dtype="string")
risk_type = keras.Input(shape=(1,), name="risk_type", dtype="string")
client_type = keras.Input(shape=(1,), name="client_type", dtype="string")
renewal_frequency = keras.Input(shape=(1,), name="renewal_frequency", dtype="string")
primary_cause = keras.Input(shape=(1,), name="primary_cause", dtype="string")
secondary_cause = keras.Input(shape=(1,), name="secondary_cause", dtype="string")
branch = keras.Input(shape=(1,), name="branch", dtype="string")


all_inputs = [
    sum_insured,
    product,
    agent,
    class_of_business,
    risk_type,
    client_type,
    renewal_frequency,
    primary_cause,
    secondary_cause,
    branch
]

sum_insured_enc = encode_numerical_feature(sum_insured, 'sum_insured', x_train)
product_enc = encode_categorical_feature(product, 'product', x_train, True)

agent_enc = encode_categorical_feature(agent, 'agent', x_train, True)
class_of_business_enc = encode_categorical_feature(class_of_business, 'class_of_business', x_train, True)
risk_type_enc = encode_categorical_feature(risk_type, 'risk_type', x_train, True)
client_type_enc = encode_categorical_feature(client_type, 'client_type', x_train, True)
renewal_frequence_enc = encode_categorical_feature(renewal_frequency, 'renewal_frequency', x_train, True)
primary_cause_enc = encode_categorical_feature(primary_cause, 'primary_cause', x_train, True)
secondary_cause_enc = encode_categorical_feature(secondary_cause, 'secondary_cause', x_train, True)
branch_enc = encode_categorical_feature(branch, 'branch', x_train, True)
x_train
all_features = keras.layers.concatenate([
    sum_insured_enc,
    product_enc,
    agent_enc,
    class_of_business_enc,
    risk_type_enc,
    client_type_enc,
    renewal_frequence_enc,
    primary_cause_enc,
    secondary_cause_enc,
    branch_enc
])


vocab_size = 50
embed_dim = 32
num_heads = 2
ff_dim = 32



embedding_layer = TokenAndPositionEmbedding(10, vocab_size, embed_dim)
x = embedding_layer(all_features)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = keras.layers.GlobalAveragePooling1D()(x)
x = keras.layers.Dropout(0.1)(x)
outputs = keras.layers.Dense(maxlen)(x)

model = keras.Model(all_inputs, outputs)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

d_model = 128
learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,epsilon=1e-9)

model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(x_train, epochs=5, validation_data=(val_dataset))


sample = {
    "sum_insured": 300000.0,
    "product": 'prod00027',
    "agent": 'ag00068',
    "class_of_business": 'cob00031',
    "risk_type": 'rt00006',
    "client_type": 'ct0003',
    "renewal_frequency": 'rf0001',
    "primary_cause": 'pc0007',
    "secondary_cause": 'sc00022',
    "branch": 'br00006'
}

input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
predictions = model.predict(input_dict)

for c in predictions[0]:
    print(c)
    print('\n')

    print(c * 7868590.62)
    print('\n')