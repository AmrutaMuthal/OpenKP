import tensorflow as tf
from transformers import BertTokenizer, TFBertModel, BertConfig

def get_model(max_len,max_kp,n1,n2):
    encoder = TFBertModel.from_pretrained("bert-base-uncased")
    input_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32)
    attention_mask = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32)
    embedding = encoder(input_ids, attention_mask=attention_mask)[0]

    bilstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n1,
                                                                     kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.00,stddev=0.15),
                                                                 dropout = 0.15,
                                                                 return_sequences=True),
                                                                 merge_mode=None)(embedding)
    pos_mask = tf.keras.layers.Input(shape=(2,max_kp),dtype='int32')
    mask_start = pos_mask[0][0]
    mask_end = pos_mask[0][1]

    start_rep_fr = tf.gather(bilstm1[0],mask_start,axis=1)
    start_rep_bk = tf.gather(bilstm1[1],mask_start,axis=1)
    end_rep_fr = tf.gather(bilstm1[0],mask_end,axis=1)
    end_rep_bk = tf.gather(bilstm1[0],mask_end,axis=1)


    span_fe_diff_fr = start_rep_fr-end_rep_fr
    span_fe_prod_fr = tf.math.multiply(start_rep_fr,end_rep_fr)
    span_fe_diff_bk = start_rep_bk-end_rep_bk
    span_fe_prod_bk = tf.math.multiply(start_rep_bk,end_rep_bk)


    span_fe = tf.keras.layers.concatenate([start_rep_fr,
                         end_rep_fr,
                         start_rep_bk,
                         end_rep_bk,
                         span_fe_diff_fr,
                         span_fe_diff_bk,
                         span_fe_prod_fr,
                         span_fe_prod_bk
                        ],2)

    
    bilstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(6,return_sequences=True,dropout = 0.25,
                                                                #kernel_initializer=tf.keras.initializers.(mean=0.0,stddev=0.05),
                                                                ),

                                             merge_mode='concat',
                                             input_shape=(max_kp,30*4))(span_fe)
    output = tf.keras.layers.Dense(1,activation='sigmoid')(bilstm2)
    
    
    kpe_model = tf.keras.models.Model(inputs=[input_ids,attention_mask,pos_mask], outputs=output)
    kpe_model.layers[3].trainable = False
    
    return kpe_model