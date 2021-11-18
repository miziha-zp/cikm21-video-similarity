import tensorflow as tf
from tensorflow.python.keras.models import Model
from transformers import TFBertModel, create_optimizer

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.position_embeddings = tf.keras.layers.Embedding(input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        # The inputs are of shape: `(batch_size, frames, num_features)`
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

    
class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, num_trans_layers, trans_dropout, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.num_layers = num_trans_layers
        self.attention = [tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim//num_heads, dropout=trans_dropout) for _ in range(self.num_layers)]
        self.dense_proj_1 = [tf.keras.layers.Dense(dense_dim, activation=tf.nn.relu) for _ in range(self.num_layers)]
        self.dense_proj_2 = [tf.keras.layers.Dense(embed_dim)  for _ in range(self.num_layers)]
        self.layernorm_1 = [tf.keras.layers.LayerNormalization() for _ in range(self.num_layers)]
        self.layernorm_2 = [tf.keras.layers.LayerNormalization() for _ in range(self.num_layers)]

    def call(self, inputs, mask):
        encoder = inputs
        for layer_index in range(self.num_layers):
            attention_output = self.attention[layer_index](encoder, encoder, encoder, attention_mask=mask)
            proj_input = self.layernorm_1[layer_index](encoder + attention_output)
            proj_output = self.dense_proj_1[layer_index](proj_input)
            proj_output = self.dense_proj_2[layer_index](proj_output)
            encoder = self.layernorm_2[layer_index](proj_input + proj_output)
        return encoder
    
class trans_M(tf.keras.layers.Layer):
    def __init__(self, embed_dim, maxlen, num_heads, num_trans_layers, trans_dense_dim, trans_dropout, **kwargs):
        super(trans_M, self).__init__(**kwargs)
        self.PosEmbedding_layer = PositionalEmbedding(maxlen, embed_dim)
        self.TransformerEncoder_layer = TransformerEncoder(embed_dim, trans_dense_dim, num_heads, num_trans_layers, trans_dropout)

    def call(self, inputs, mask, **kwargs):
        x = self.PosEmbedding_layer(inputs)
        v_emb = self.TransformerEncoder_layer(x, mask)
        return v_emb
    
class LinearTransLayer(tf.keras.layers.Layer):

    def __init__(self, embdim, trainable=True, **kwargs):
        self._embdim = embdim
        self._trainable = trainable
        super(LinearTransLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self._weights_lt = self.add_weight(
            shape=(input_shape[-1], self._embdim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name = 'weights_lt')
        self._bias_lt = self.add_weight(
            shape=(self._embdim, ),
            initializer='Zeros',
            trainable=self._trainable,
            name = 'bias_lt')
        super(LinearTransLayer, self).build(input_shape)

    def call(self, X):
        outputs = tf.nn.bias_add(tf.matmul(X, self._weights_lt), self._bias_lt)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

class SENet(tf.keras.layers.Layer):
    def __init__(self, channels, ratio=8, **kwargs):
        super(SENet, self).__init__(**kwargs)
        self.fc = tf.keras.Sequential([
            tf.keras.layers.Dense(channels // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False),
            tf.keras.layers.Dense(channels, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)
        ])

    def call(self, inputs, **kwargs):
        se = self.fc(inputs)
        outputs = tf.math.multiply(inputs, se)
        return outputs

class ConcatDenseSE(tf.keras.layers.Layer):
    """Fusion using Concate + Dense + SENet"""

    def __init__(self, hidden_size, se_ratio, **kwargs):
        super().__init__(**kwargs)
        self.fusion = tf.keras.layers.Dense(hidden_size*2, activation='relu')
        self.fusion_dropout = tf.keras.layers.Dropout(0.2)
        self.enhance = SENet(channels=hidden_size*2, ratio=se_ratio)

    def call(self, inputs, **kwargs):
#         embeddings = tf.concat(inputs, axis=1)
        embeddings = self.fusion_dropout(inputs)
        embedding = self.fusion(embeddings)
        embedding = self.enhance(embedding)

        return embedding
    
class MultiModal(Model):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.titleasr_bert = TFBertModel.from_pretrained(config.bert_dir)
        
        self.lts_1 = LinearTransLayer(768)
        self.lts_2 = LinearTransLayer(768)
        
        self.trans_vis = trans_M(config.all_embedding_size, config.max_frames, config.vis_trans_num_heads, config.vis_num_trans_layers, config.vis_trans_dense_dim, config.vis_trans_dropout)
        
        self.trans_all = trans_M(config.all_embedding_size, config.all_maxlen, config.all_trans_num_heads, config.all_num_trans_layers, config.all_trans_dense_dim, config.all_trans_dropout)
        
        self.vec_vision_pooling_layer = tf.keras.layers.GlobalAveragePooling1D()
        self.num_labels = config.num_labels
        self.fusion = ConcatDenseSE(config.hidden_size, config.ratio)
        self.classifier = tf.keras.layers.Dense(self.num_labels, activation='sigmoid')
        self.project_1 = tf.keras.layers.Dense(512, activation='relu')
        self.BN_1 = tf.keras.layers.BatchNormalization()
        self.project_2 = tf.keras.layers.Dense(256, activation=None)
        self.titleasr_bert_optimizer, self.titleasr_bert_lr = create_optimizer(init_lr=config.bert_lr,
                                                             num_train_steps=config.bert_total_steps,
                                                             num_warmup_steps=config.bert_warmup_steps)
        self.optimizer, self.lr = create_optimizer(init_lr=config.lr,
                                                   num_train_steps=config.total_steps,
                                                   num_warmup_steps=config.warmup_steps)
        
        self.titleasr_bert_variables, self.titleasr_num_bert, self.all_variables = None, None, None
    
    def call(self, inputs, **kwargs):
        bert_embedding_title = self.titleasr_bert([inputs['input_ids_title'], inputs['mask_title']])[0] # out -> (256, 32, 768)
        bert_embedding_asr = self.titleasr_bert([inputs['input_ids_asr'], inputs['mask_asr']])[0] # out -> (256, 32, 768) 
        bert_embedding_title_mask = inputs['mask_title'] # (256,32)
        bert_embedding_asr_mask = inputs['mask_asr']# (256,32)
        bert_embedding_title_mask = bert_embedding_title_mask[:, tf.newaxis,:]
        bert_embedding_asr_mask = bert_embedding_asr_mask[:, tf.newaxis,:]
        vision_mask = tf.cast(tf.reduce_any(tf.cast(inputs['frames'], "bool"), axis=-1),"int32") # (256,32) 全1.0
        vision_mask = vision_mask[:, tf.newaxis,:]
        vision_embedding_linear_0 = self.lts_1(inputs['frames']) # (256, 32, 768)
        vision_embedding = self.trans_vis(vision_embedding_linear_0,vision_mask) # out -> (256, 32, 768)
        vision_embedding_linear_1 = self.lts_2(vision_embedding) # (256, 32, 768)     
        
        bert_embedding_all = tf.concat([bert_embedding_title,bert_embedding_asr,vision_embedding_linear_1], axis=1) # (256, 96, 768)
        mask_all = tf.concat([bert_embedding_title_mask,bert_embedding_asr_mask,vision_mask], axis=-1)# (256,1,96)
        
        all_fea = self.trans_all(bert_embedding_all,mask_all) # (256, 96, 768)
        cls_title = all_fea[:,0,:]# (256,768)
        cls_asr = all_fea[:,32,:]# (256,768)
        vec_vision = all_fea[:,64:,:]# (256,32,768)
        vec_vision_pooling = self.vec_vision_pooling_layer(vec_vision)# (256,768)
        final_embedding = tf.keras.layers.concatenate([cls_title,cls_asr,vec_vision_pooling])# (256,768*3)
        final_embedding_enhance = self.fusion(final_embedding)# (256,256*2)
        predictions = self.classifier(final_embedding_enhance)
        emb_1 = self.project_1(final_embedding)
        emb_2 = self.BN_1(emb_1)
        emb_3 = self.project_2(emb_2)
        return predictions,  final_embedding_enhance, cls_title, cls_asr, vec_vision_pooling, emb_3

    def get_variables(self):
        if not self.all_variables:  # is None, not initialized
            self.titleasr_bert_variables = self.titleasr_bert.trainable_variables
            self.titleasr_num_bert = len(self.titleasr_bert_variables)
            
            self.normal_variables = self.lts_1.trainable_variables + self.lts_2.trainable_variables + \
                                    self.trans_vis.trainable_variables + \
                                    self.trans_all.trainable_variables + self.vec_vision_pooling_layer.trainable_variables + \
                                    self.fusion.trainable_variables + self.classifier.trainable_variables+\
                                    self.project_1.trainable_variables + self.BN_1.trainable_variables + self.project_2.trainable_variables
            self.all_variables = self.titleasr_bert_variables + self.normal_variables
        print('----------|len(self.all_variables):',len(self.all_variables),'|----------')
        return self.all_variables

    def optimize(self, gradients):
        
        title_bert_gradients = gradients[:self.titleasr_num_bert]
        self.titleasr_bert_optimizer.apply_gradients(zip(title_bert_gradients, self.titleasr_bert_variables))
        
        normal_gradients = gradients[self.titleasr_num_bert:]
        self.optimizer.apply_gradients(zip(normal_gradients, self.normal_variables))
