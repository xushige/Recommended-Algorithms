import tensorflow as tf
from typing import OrderedDict

'''逻辑回归 (LR)'''
class LogisticalRegression(tf.keras.Model):
    def __init__(self, emb_dim, item_feature_category_list, profile_feature_category_list, embedding_feature=False) -> None:
        super().__init__()
        '''
            emb_dim: embedding维度
            item_feature_category_list: 商品特征类别列表
            profile_feature_category_list: 用户特征类别列表
            embedding_feature: 输入feature是否为embedding特征
        '''
        # embedding层
        self.embedding_layer = OrderedDict() 
        for i in range(len(item_feature_category_list)):
            self.embedding_layer['ItemFc_%d'%(i)] = tf.keras.layers.Embedding(item_feature_category_list[i], emb_dim)
        for i in range(len(profile_feature_category_list)):
            self.embedding_layer['ProfileFc_%d'%(i)] = tf.keras.layers.Embedding(profile_feature_category_list[i], emb_dim)
        # l1 正则方便特征筛选
        self.lr = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l1(0.001), activation='sigmoid') 
        self.flatten = tf.keras.layers.Flatten()
        self.embedding_feature = embedding_feature

    def call(self, target_ad, ubs_feature, profile_feature, context_feature):
        '''
            target_ad: 待预测商品特征张量 (sparse特征) [Batch_size, Feature_num]
            ubs_feature: 用户行为序列特征张量 (sparse特征) [Batch_size, UBS_leng, Feature_num]
            profile_feature: 用户特征张量 (sparse特征) [Batch_size, Feature_num]
            target_ad: 上下文特征张量 (dense特征) [Batch_size, Feature_num]
        '''
        # 特征 Embedding 化
        if not self.embedding_feature:
            target_ad = tf.concat([self.embedding_layer['ItemFc_%d'%(i)](target_ad[:, i:i+1]) for i in range(target_ad.shape[-1])], axis=-2)
            ubs_feature = tf.concat([self.embedding_layer['ItemFc_%d'%(i)](ubs_feature[:, :, i:i+1]) for i in range(ubs_feature.shape[-1])], axis=-2)
            profile_feature = tf.concat([self.embedding_layer['ProfileFc_%d'%(i)](profile_feature[:, i:i+1]) for i in range(profile_feature.shape[-1])], axis=-2)
        # ubs序列average处理
        ubs_feature = tf.reduce_mean(ubs_feature, axis=1)
        # 特征flatten & concat
        total_emb = tf.concat([
            self.flatten(target_ad),
            self.flatten(ubs_feature),
            self.flatten(profile_feature),
            context_feature
        ], axis=-1)
        out = self.lr(total_emb)
        return out

'''因子分解机 (FM)'''
class FactorizationMachine(tf.keras.Model):
    def __init__(self, emb_dim, item_feature_category_list, profile_feature_category_list, embedding_feature=False) -> None:
        super().__init__()
        '''
            emb_dim: embedding维度
            item_feature_category_list: 商品特征类别列表
            profile_feature_category_list: 用户特征类别列表
            embedding_feature: 输入feature是否为embedding特征
        '''
        self.embedding_layer = OrderedDict()
        for i in range(len(item_feature_category_list)):
            self.embedding_layer['ItemFc_%d'%(i)] = tf.keras.layers.Embedding(item_feature_category_list[i], emb_dim)
        for i in range(len(profile_feature_category_list)):
            self.embedding_layer['ProfileFc_%d'%(i)] = tf.keras.layers.Embedding(profile_feature_category_list[i], emb_dim)
        self.lr = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l1(0.0001), activation='sigmoid') 
        self.flatten = tf.keras.layers.Flatten()
        self.sigmoid = tf.keras.layers.Activation('sigmoid')
        self.embedding_feature = embedding_feature

    def call(self, target_ad, ubs_feature, profile_feature, context_feature):
        '''
            target_ad: 待预测商品特征张量 (sparse特征) [Batch_size, Feature_num]
            ubs_feature: 用户行为序列特征张量 (sparse特征) [Batch_size, UBS_leng, Feature_num]
            profile_feature: 用户特征张量 (sparse特征) [Batch_size, Feature_num]
            target_ad: 上下文特征张量 (dense特征) [Batch_size, Feature_num]
        '''
        # 特征 Embedding 化
        if not self.embedding_feature:
            target_ad = tf.concat([self.embedding_layer['ItemFc_%d'%(i)](target_ad[:, i:i+1]) for i in range(target_ad.shape[-1])], axis=-2)
            ubs_feature = tf.concat([self.embedding_layer['ItemFc_%d'%(i)](ubs_feature[:, :, i:i+1]) for i in range(ubs_feature.shape[-1])], axis=-2)
            profile_feature = tf.concat([self.embedding_layer['ProfileFc_%d'%(i)](profile_feature[:, i:i+1]) for i in range(profile_feature.shape[-1])], axis=-2)
        # ubs序列average处理
        ubs_feature = tf.reduce_mean(ubs_feature, axis=1)
        # 类别型特征embedding
        sparse_emb = tf.concat([target_ad, ubs_feature, profile_feature], axis=-2) 
        # Cross out 计算：（和的平方-平方的和）/ 2
        square_sum = tf.reduce_sum(tf.square(tf.reduce_sum(sparse_emb, axis=1)), axis=1)
        sum_square = tf.reduce_sum(tf.reduce_sum(tf.square(sparse_emb), axis=1), axis=1)
        cross_out = tf.expand_dims(self.sigmoid((square_sum-sum_square)/2), axis=-1)
        # LR out 计算
        lr_out = self.lr(tf.concat([
            self.flatten(sparse_emb),
            context_feature
        ], axis=-1))
        # FM out
        fm_out = (lr_out + cross_out) / 2
        return fm_out

'''Deep Neural Network (DNN)'''
class DeepNeuralNetwork(tf.keras.Model):
    def __init__(self, emb_dim, item_feature_category_list, profile_feature_category_list, embedding_feature=False) -> None:
        super().__init__()
        '''
            emb_dim: embedding维度
            item_feature_category_list: 商品特征类别列表
            profile_feature_category_list: 用户特征类别列表
            embedding_feature: 输入feature是否为embedding特征
        '''
        # Embedding layer
        self.embedding_layer = OrderedDict()
        for i in range(len(item_feature_category_list)):
            self.embedding_layer['ItemFc_%d'%(i)] = tf.keras.layers.Embedding(item_feature_category_list[i], emb_dim)
        for i in range(len(profile_feature_category_list)):
            self.embedding_layer['ProfileFc_%d'%(i)] = tf.keras.layers.Embedding(profile_feature_category_list[i], emb_dim)
        # Deep Neural Network
        self.dense_tower = tf.keras.Sequential([
            tf.keras.layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l1(0.001)), # 第一次l1正则筛选特征，隐层l2正则
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.flatten = tf.keras.layers.Flatten()
        self.embedding_feature = embedding_feature

    def call(self, target_ad, ubs_feature, profile_feature, context_feature):
        '''
            target_ad: 待预测商品特征张量 (sparse特征) [Batch_size, Feature_num]
            ubs_feature: 用户行为序列特征张量 (sparse特征) [Batch_size, UBS_leng, Feature_num]
            profile_feature: 用户特征张量 (sparse特征) [Batch_size, Feature_num]
            target_ad: 上下文特征张量 (dense特征) [Batch_size, Feature_num]
        '''
        # 特征 Embedding 化
        if not self.embedding_feature:
            target_ad = tf.concat([self.embedding_layer['ItemFc_%d'%(i)](target_ad[:, i:i+1]) for i in range(target_ad.shape[-1])], axis=-2)
            ubs_feature = tf.concat([self.embedding_layer['ItemFc_%d'%(i)](ubs_feature[:, :, i:i+1]) for i in range(ubs_feature.shape[-1])], axis=-2)
            profile_feature = tf.concat([self.embedding_layer['ProfileFc_%d'%(i)](profile_feature[:, i:i+1]) for i in range(profile_feature.shape[-1])], axis=-2)
        # ubs序列average处理
        ubs_feature = tf.reduce_mean(ubs_feature, axis=1)
        # embedding concat
        total_emb = tf.concat([
            self.flatten(target_ad),
            self.flatten(ubs_feature),
            self.flatten(profile_feature),
            context_feature
        ], axis=-1)
        out = self.dense_tower(total_emb)
        return out

'''Wide & Deep'''
class WideAndDeep(tf.keras.Model):
    def __init__(self, emb_dim, item_feature_category_list, profile_feature_category_list, embedding_feature=False) -> None:
        super().__init__()
        '''
            emb_dim: embedding维度
            item_feature_category_list: 商品特征类别列表
            profile_feature_category_list: 用户特征类别列表
            embedding_feature: 输入feature是否为embedding特征
        '''
        # embedding_layer
        self.embedding_layer = OrderedDict()
        for i in range(len(item_feature_category_list)):
            self.embedding_layer['ItemFc_%d' % (i)] = tf.keras.layers.Embedding(item_feature_category_list[i], emb_dim)
        for i in range(len(profile_feature_category_list)):
            self.embedding_layer['ProfileFc_%d' % (i)] = tf.keras.layers.Embedding(profile_feature_category_list[i], emb_dim)
        # Wide 部分，L1正则
        self.lr = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l1(0.001))
        # Deep 部分 L2正则
        self.dense_tower = tf.keras.Sequential([
            tf.keras.layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l1(0.001)), # 第一次l1正则筛选特征，隐层l2正则
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.flatten = tf.keras.layers.Flatten()
        self.embedding_feature = embedding_feature

    def call(self, target_ad, ubs_feature, profile_feature, context_feature):
        '''
            target_ad: 待预测商品特征张量 (sparse特征) [Batch_size, Feature_num]
            ubs_feature: 用户行为序列特征张量 (sparse特征) [Batch_size, UBS_leng, Feature_num]
            profile_feature: 用户特征张量 (sparse特征) [Batch_size, Feature_num]
            target_ad: 上下文特征张量 (dense特征) [Batch_size, Feature_num]
        '''
        # 特征 Embedding 化
        if not self.embedding_feature:
            target_ad = tf.concat([self.embedding_layer['ItemFc_%d'%(i)](target_ad[:, i:i+1]) for i in range(target_ad.shape[-1])], axis=-2)
            ubs_feature = tf.concat([self.embedding_layer['ItemFc_%d'%(i)](ubs_feature[:, :, i:i+1]) for i in range(ubs_feature.shape[-1])], axis=-2)
            profile_feature = tf.concat([self.embedding_layer['ProfileFc_%d'%(i)](profile_feature[:, i:i+1]) for i in range(profile_feature.shape[-1])], axis=-2)
        # ubs序列average处理
        ubs_feature = tf.reduce_mean(ubs_feature, axis=1)
        # 特征concat
        total_embedding = tf.concat([
            self.flatten(target_ad),
            self.flatten(ubs_feature),
            self.flatten(profile_feature),
            context_feature
        ], axis=-1)
        # 输出
        wide_out = self.lr(total_embedding)
        deep_out = self.dense_tower(total_embedding)
        out = (wide_out + deep_out) / 2
        return out

'''DeepFM'''
class DeepFM(tf.keras.Model):
    def __init__(self, emb_dim, item_feature_category_list, profile_feature_category_list, embedding_feature=False) -> None:
        super().__init__()
        '''
            emb_dim: embedding维度
            item_feature_category_list: 商品特征类别列表
            profile_feature_category_list: 用户特征类别列表
            embedding_feature: 输入feature是否为embedding特征
        '''
        # embedding_layer
        self.embedding_layer = OrderedDict()
        for i in range(len(item_feature_category_list)):
            self.embedding_layer['ItemFc_%d'%(i)] = tf.keras.layers.Embedding(item_feature_category_list[i], emb_dim)
        for i in range(len(profile_feature_category_list)):
            self.embedding_layer['ProfileFc_%d'%(i)] = tf.keras.layers.Embedding(profile_feature_category_list[i], emb_dim)
        # FM part
        self.lr = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l1(0.001))
        # Deep part
        self.dense_tower = tf.keras.Sequential([
            tf.keras.layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l1(0.001)), # 第一次l1正则筛选特征，隐层l2正则
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.flatten = tf.keras.layers.Flatten()
        self.sigmoid = tf.keras.layers.Activation('sigmoid')
        self.embedding_feature = embedding_feature

    def call(self, target_ad, ubs_feature, profile_feature, context_feature):
        '''
            target_ad: 待预测商品特征张量 (sparse特征) [Batch_size, Feature_num]
            ubs_feature: 用户行为序列特征张量 (sparse特征) [Batch_size, UBS_leng, Feature_num]
            profile_feature: 用户特征张量 (sparse特征) [Batch_size, Feature_num]
            target_ad: 上下文特征张量 (dense特征) [Batch_size, Feature_num]
        '''
        # 特征 Embedding 化
        if not self.embedding_feature:
            target_ad = tf.concat([self.embedding_layer['ItemFc_%d'%(i)](target_ad[:, i:i+1]) for i in range(target_ad.shape[-1])], axis=-2)
            ubs_feature = tf.concat([self.embedding_layer['ItemFc_%d'%(i)](ubs_feature[:, :, i:i+1]) for i in range(ubs_feature.shape[-1])], axis=-2)
            profile_feature = tf.concat([self.embedding_layer['ProfileFc_%d'%(i)](profile_feature[:, i:i+1]) for i in range(profile_feature.shape[-1])], axis=-2)
        # ubs序列average处理
        ubs_feature = tf.reduce_mean(ubs_feature, axis=1)
        # 离散特征concat
        sparse_fc_embedding = tf.concat([
            target_ad,
            ubs_feature,
            profile_feature
        ], axis=-2)
        # 所有特征concat
        total_fc_embedding = tf.concat([
            self.flatten(sparse_fc_embedding),
            context_feature
        ], axis=-1)
        # FM part
        lr_out = self.lr(total_fc_embedding)
        square_sum = tf.reduce_sum(tf.square(tf.reduce_sum(sparse_fc_embedding, axis=-2)), axis=-1)
        sum_square = tf.reduce_sum(tf.reduce_sum(tf.square(sparse_fc_embedding), axis=-2), axis=-1)
        cross_out = tf.expand_dims(self.sigmoid((square_sum-sum_square)/2), axis=-1)
        fm_out = (cross_out + lr_out) / 2
        # Deep part
        deep_out = self.dense_tower(total_fc_embedding)
        # output
        out = (fm_out + deep_out) / 2
        return out

'''Deep & Cross Network (DCN)'''
class DeepCrossNetwork(tf.keras.Model):
    def __init__(self, emb_dim, item_feature_category_list, profile_feature_category_list, embedding_feature=False, cross_layer_num=4, sparse_feature_num=50) -> None:
        super().__init__()
        '''
            emb_dim: embedding维度
            item_feature_category_list: 商品特征类别列表
            profile_feature_category_list: 用户特征类别列表
            embedding_feature: 输入feature是否为embedding特征
            cross_layer_num: cross显式交叉层数
            sparse_feature_num: 用于交叉的“离散特征数目”
        '''
        # embedding layer
        self.embedding_layer = OrderedDict()
        for i in range(len(item_feature_category_list)):
            self.embedding_layer['ItemFc_%d'%(i)] = tf.keras.layers.Embedding(item_feature_category_list[i], emb_dim)
        for i in range(len(profile_feature_category_list)):
            self.embedding_layer['ProfileFc_%d'%(i)] = tf.keras.layers.Embedding(profile_feature_category_list[i], emb_dim)
        # cross layer
        self.cross_layer_num = cross_layer_num
        self.cross_layer = OrderedDict()
        for i in range(self.cross_layer_num):
            self.cross_layer['cross_%d'%(i)] = tf.keras.layers.Dense(sparse_feature_num, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        # deep layer
        self.deep_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(sparse_feature_num, kernel_regularizer=tf.keras.regularizers.l1(0.001)), # 第一次l1正则筛选特征，隐层l2正则
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(sparse_feature_num, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(sparse_feature_num, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('relu')
        ])
        self.lr = tf.keras.layers.Dense(1, activation='sigmoid')
        self.flatten = tf.keras.layers.Flatten()
        self.embedding_feature = embedding_feature

    def call(self, target_ad, ubs_feature, profile_feature, context_feature):
        '''
            target_ad: 待预测商品特征张量 (sparse特征) [Batch_size, Feature_num]
            ubs_feature: 用户行为序列特征张量 (sparse特征) [Batch_size, UBS_leng, Feature_num]
            profile_feature: 用户特征张量 (sparse特征) [Batch_size, Feature_num]
            target_ad: 上下文特征张量 (dense特征) [Batch_size, Feature_num]
        '''
        # 特征 Embedding 化
        if not self.embedding_feature:
            target_ad = tf.concat([self.embedding_layer['ItemFc_%d'%(i)](target_ad[:, i:i+1]) for i in range(target_ad.shape[-1])], axis=-2)
            ubs_feature = tf.concat([self.embedding_layer['ItemFc_%d'%(i)](ubs_feature[:, :, i:i+1]) for i in range(ubs_feature.shape[-1])], axis=-2)
            profile_feature = tf.concat([self.embedding_layer['ProfileFc_%d'%(i)](profile_feature[:, i:i+1]) for i in range(profile_feature.shape[-1])], axis=-2)
        # ubs序列average处理
        ubs_feature = tf.reduce_mean(ubs_feature, axis=1)
        # cross 特征显式交叉
        x0 = tf.concat([
            target_ad,
            ubs_feature,
            profile_feature
        ], axis=-2)
        cross_out = tf.tile(x0, [1, 1, 1])
        for i in range(self.cross_layer_num):
            cross_out = tf.transpose(self.cross_layer['cross_%d'%(i)](tf.matmul(tf.transpose(x0, [0, 2, 1]), cross_out)), [0, 2, 1]) + cross_out
        # deep 特征隐式交叉
        deep_out = self.deep_layer(tf.transpose(x0, [0, 2, 1]))
        # LR 分类
        total_embedding = tf.concat([
            self.flatten(cross_out),
            self.flatten(deep_out),
            context_feature
        ], axis=-1)
        out = self.lr(total_embedding)
        return out

'''Deep Interest Network (DIN)'''
class DeepIntersetNetwork(tf.keras.Model):
    def __init__(self, emb_dim, item_feature_category_list, profile_feature_category_list, embedding_feature=False, ubs_leng=100) -> None:
        super().__init__()
        '''
            emb_dim: embedding维度
            item_feature_category_list: 商品特征类别列表
            profile_feature_category_list: 用户特征类别列表
            embedding_feature: 输入feature是否为embedding特征
            ubs_leng: ubs序列长度
        '''
        # Embedding layer
        self.embedding_layer = OrderedDict()
        for i in range(len(item_feature_category_list)):
            self.embedding_layer['ItemFc_%d'%(i)] = tf.keras.layers.Embedding(item_feature_category_list[i], emb_dim)
        for i in range(len(profile_feature_category_list)):
            self.embedding_layer['ProfileFc_%d'%(i)] = tf.keras.layers.Embedding(profile_feature_category_list[i], emb_dim)
        # attention layer, ubs_leng表示ubs序列长度
        self.din_att_fc = OrderedDict()
        for i in range(ubs_leng):
            self.din_att_fc['att_%d'%(i)] = tf.keras.layers.Dense(1, activation='sigmoid')
        # dense塔
        self.dense_tower = tf.keras.Sequential([
            tf.keras.layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l1(0.001)), # 第一次l1正则筛选特征，隐层l2正则
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.flatten = tf.keras.layers.Flatten()
        self.embedding_feature = embedding_feature

    def call(self, target_ad, ubs_feature, profile_feature, context_feature):
        '''
            target_ad: 待预测商品特征张量 (sparse特征) [Batch_size, Feature_num]
            ubs_feature: 用户行为序列特征张量 (sparse特征) [Batch_size, UBS_leng, Feature_num]
            profile_feature: 用户特征张量 (sparse特征) [Batch_size, Feature_num]
            target_ad: 上下文特征张量 (dense特征) [Batch_size, Feature_num]
        '''
        # 特征 Embedding 化
        if not self.embedding_feature:
            ubs_feature = tf.concat([self.embedding_layer['ItemFc_%d'%(i)](ubs_feature[:, :, i:i+1]) for i in range(ubs_feature.shape[-1])], axis=-2)
            profile_feature = tf.concat([self.embedding_layer['ProfileFc_%d'%(i)](profile_feature[:, i:i+1]) for i in range(profile_feature.shape[-1])], axis=-2)
            target_ad = tf.concat([self.embedding_layer['ItemFc_%d'%(i)](target_ad[:, i:i+1]) for i in range(target_ad.shape[-1])], axis=-2)
        # ubs中每一个用户行为x_i与target融合后进行attention权重计算，融合方式[x_i, target, x_i+target, x_i*target]
        att_vector = tf.concat([
                self.din_att_fc['att_%d'%(i)](
                    tf.concat([
                        self.flatten(ubs_feature[:, i]),
                        self.flatten(target_ad),
                        self.flatten(ubs_feature[:, i]) * self.flatten(target_ad),
                        self.flatten(ubs_feature[:, i]) + self.flatten(target_ad)
                    ], axis=-1)
                ) for i in range(ubs_feature.shape[1])
            ], axis=-1)
        # attention对ubs序列加权
        att_vector = tf.expand_dims(tf.expand_dims(att_vector, -1), -1)
        ubs_feature = tf.reduce_mean(ubs_feature * att_vector, axis=1)
        # 特征Concat后经过Dense塔分类
        total_embedding = tf.concat([
            self.flatten(ubs_feature),
            self.flatten(target_ad),
            self.flatten(profile_feature),
            context_feature
        ], axis=-1)
        out = self.dense_tower(total_embedding)
        return out

'''Transformer'''
class Transformer(tf.keras.Model):
    def __init__(self, emb_dim, item_feature_category_list, profile_feature_category_list, embedding_feature=False, ubs_leng=100, att_dim=64, head_num=4) -> None:
        super().__init__()
        '''
            emb_dim: embedding维度
            item_feature_category_list: 商品特征类别列表
            profile_feature_category_list: 用户特征类别列表
            embedding_feature: 输入feature是否为embedding特征
            ubs_leng: ubs序列长度
            att_dim: QKV自注意力矩阵维度
            head_num: 多头自注意力数
        '''
        self.att_dim = att_dim
        self.head_num = head_num
        # Embedding layer
        self.embedding_layer = OrderedDict()
        for i in range(len(item_feature_category_list)):
            self.embedding_layer['ItemFc_%d'%(i)] = tf.keras.layers.Embedding(item_feature_category_list[i], emb_dim)
        for i in range(len(profile_feature_category_list)):
            self.embedding_layer['ProfileFc_%d'%(i)] = tf.keras.layers.Embedding(profile_feature_category_list[i], emb_dim)
        # Multi-head Self-Attention layer
        self.q, self.k, self.v, self.linear = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
        for i in range(ubs_leng): 
            self.q['ItemFc_%d'%(i)] = tf.keras.layers.Dense(att_dim * head_num, use_bias=False)
            self.k['ItemFc_%d'%(i)] = tf.keras.layers.Dense(att_dim * head_num, use_bias=False)
            self.v['ItemFc_%d'%(i)] = tf.keras.layers.Dense(att_dim * head_num, use_bias=False)
            self.linear['ItemFc_%d'%(i)] = tf.keras.layers.Dense(emb_dim)
        # Dense 塔
        self.dense_tower = tf.keras.Sequential([
            tf.keras.layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l1(0.001)), # 第一次l1正则筛选特征，隐层l2正则
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.flatten = tf.keras.layers.Flatten()
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.relu = tf.keras.layers.Activation('relu')
        self.embedding_feature = embedding_feature

    def call(self, target_ad, ubs_feature, profile_feature, context_feature):
        '''
            target_ad: 待预测商品特征张量 (sparse特征) [Batch_size, Feature_num]
            ubs_feature: 用户行为序列特征张量 (sparse特征) [Batch_size, UBS_leng, Feature_num]
            profile_feature: 用户特征张量 (sparse特征) [Batch_size, Feature_num]
            target_ad: 上下文特征张量 (dense特征) [Batch_size, Feature_num]
        '''
        # 特征 Embedding 化
        if not self.embedding_feature:
            target_ad = tf.concat([self.embedding_layer['ItemFc_%d'%(i)](target_ad[:, i:i+1]) for i in range(target_ad.shape[-1])], axis=-2)
            ubs_feature = tf.concat([self.embedding_layer['ItemFc_%d'%(i)](ubs_feature[:, :, i:i+1]) for i in range(ubs_feature.shape[-1])], axis=-2)
            profile_feature = tf.concat([self.embedding_layer['ProfileFc_%d'%(i)](profile_feature[:, i:i+1]) for i in range(profile_feature.shape[-1])], axis=-2)
        # ubs每一个用户历史行为进行 Multi-head Self-Attention, 而后mean
        ubs_feature_tensor = tf.zeros_like(target_ad)
        for i in range(ubs_feature.shape[1]):
            cur_feature = ubs_feature[:, i]
            q_out = tf.transpose(tf.reshape(self.q['ItemFc_%d'%(i)](cur_feature), [-1, cur_feature.shape[1], self.head_num, self.att_dim]), [0, 2, 1, 3])
            k_out = tf.transpose(tf.reshape(self.k['ItemFc_%d'%(i)](cur_feature), [-1, cur_feature.shape[1], self.head_num, self.att_dim]), [0, 2, 1, 3])
            v_out = tf.transpose(tf.reshape(self.v['ItemFc_%d'%(i)](cur_feature), [-1, cur_feature.shape[1], self.head_num, self.att_dim]), [0, 2, 1, 3])
            # softmax(q * k / att_dim ** 0.5) * v
            att = tf.matmul(tf.nn.softmax(tf.matmul(q_out, tf.transpose(k_out, [0, 1, 3, 2])) / (self.att_dim ** 0.5), axis=-1), v_out)
            att_out = self.linear['ItemFc_%d'%(i)](tf.reshape(tf.transpose(att, [0, 2, 1, 3]), [-1, att.shape[2], self.att_dim*self.head_num]))
            ubs_feature_tensor += self.layernorm(att_out)
        ubs_feature = self.relu(ubs_feature_tensor / float(ubs_feature.shape[1]))
        # 特征concat
        total_embedding = tf.concat([
            self.flatten(ubs_feature),
            self.flatten(target_ad),
            self.flatten(profile_feature),
            context_feature
        ], axis=-1)
        # output
        out = self.dense_tower(total_embedding)
        return out
    
'''Entire Space Multi-task Model (ESMM)'''
class ESMM(tf.keras.Model):
    def __init__(self, emb_dim, item_feature_category_list, profile_feature_category_list, sub_models={'CTR':LogisticalRegression, 'CTCVR': FactorizationMachine}) -> None:
        super().__init__()
        '''
            emb_dim: embedding维度
            item_feature_category_list: 商品特征类别列表
            profile_feature_category_list: 用户特征类别列表
            sub_models: 子任务模型【CTR模型, CTCVR模型】
        '''
        # embedding_layer
        self.embedding_layer = OrderedDict()
        for i in range(len(item_feature_category_list)):
            self.embedding_layer['ItemFc_%d'%(i)] = tf.keras.layers.Embedding(item_feature_category_list[i], emb_dim)
        for i in range(len(profile_feature_category_list)):
            self.embedding_layer['ProfileFc_%d'%(i)] = tf.keras.layers.Embedding(profile_feature_category_list[i], emb_dim)
        # CTR model
        self.ctr_model = sub_models['CTR'](emb_dim, item_feature_category_list, profile_feature_category_list, embedding_feature=True)
        # CTCVR model
        self.ctcvr_model = sub_models['CTCVR'](emb_dim, item_feature_category_list, profile_feature_category_list, embedding_feature=True)
    
    def call(self, target_ad, ubs_feature, profile_feature, context_feature):
        '''
            target_ad: 待预测商品特征张量 (sparse特征) [Batch_size, Feature_num]
            ubs_feature: 用户行为序列特征张量 (sparse特征) [Batch_size, UBS_leng, Feature_num]
            profile_feature: 用户特征张量 (sparse特征) [Batch_size, Feature_num]
            target_ad: 上下文特征张量 (dense特征) [Batch_size, Feature_num]
        '''
        # 多任务共享 embedding
        target_ad = tf.concat([self.embedding_layer['ItemFc_%d'%(i)](target_ad[:, i:i+1]) for i in range(target_ad.shape[-1])], axis=-2)
        ubs_feature = tf.concat([self.embedding_layer['ItemFc_%d'%(i)](ubs_feature[:, :, i:i+1]) for i in range(ubs_feature.shape[-1])], axis=-2)
        profile_feature = tf.concat([self.embedding_layer['ProfileFc_%d'%(i)](profile_feature[:, i:i+1]) for i in range(profile_feature.shape[-1])], axis=-2)
        # CTR out
        ctr = self.ctr_model(target_ad, ubs_feature, profile_feature, context_feature)
        # CTCVR out
        ctcvr = self.ctcvr_model(target_ad, ubs_feature, profile_feature, context_feature)
        return ctr, ctcvr
    
'''Multi-gate Mixture-Of-Experts (MMOE)'''
class MMOE(tf.keras.Model):
    def __init__(self, emb_dim, item_feature_category_list, profile_feature_category_list, experts_num=3) -> None:
        super().__init__()
        '''
            emb_dim: embedding维度
            item_feature_category_list: 商品特征类别列表
            profile_feature_category_list: 用户特征类别列表
            experts_num: 专家数量
        '''
        # embedding_layer
        self.embedding_layer = OrderedDict()
        for i in range(len(item_feature_category_list)):
            self.embedding_layer['ItemFc_%d'%(i)] = tf.keras.layers.Embedding(item_feature_category_list[i], emb_dim)
        for i in range(len(profile_feature_category_list)):
            self.embedding_layer['ProfileFc_%d'%(i)] = tf.keras.layers.Embedding(profile_feature_category_list[i], emb_dim)
        # experts layer
        self.experts_layer = OrderedDict()
        for i in range(experts_num):
            self.experts_layer['expert_%d'%(i)] = tf.keras.layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l1(0.001), activation='relu')
        # task gate
        self.ctr_gate = tf.keras.layers.Dense(experts_num, kernel_regularizer=tf.keras.regularizers.l1(0.001), activation='sigmoid')
        self.ctcvr_gate = tf.keras.layers.Dense(experts_num, kernel_regularizer=tf.keras.regularizers.l1(0.001), activation='sigmoid')
        # multi-task tower
        self.ctr_tower = tf.keras.Sequential([
            tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.ctcvr_tower = tf.keras.Sequential([
            tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.flatten = tf.keras.layers.Flatten()

    def call(self, target_ad, ubs_feature, profile_feature, context_feature):
        '''
            target_ad: 待预测商品特征张量 (sparse特征) [Batch_size, Feature_num]
            ubs_feature: 用户行为序列特征张量 (sparse特征) [Batch_size, UBS_leng, Feature_num]
            profile_feature: 用户特征张量 (sparse特征) [Batch_size, Feature_num]
            target_ad: 上下文特征张量 (dense特征) [Batch_size, Feature_num]
        '''
        # 多任务共享 embedding
        target_ad = tf.concat([self.embedding_layer['ItemFc_%d'%(i)](target_ad[:, i:i+1]) for i in range(target_ad.shape[-1])], axis=-2)
        ubs_feature = tf.concat([self.embedding_layer['ItemFc_%d'%(i)](ubs_feature[:, :, i:i+1]) for i in range(ubs_feature.shape[-1])], axis=-2)
        profile_feature = tf.concat([self.embedding_layer['ProfileFc_%d'%(i)](profile_feature[:, i:i+1]) for i in range(profile_feature.shape[-1])], axis=-2)
        # ubs序列average处理
        ubs_feature = tf.reduce_mean(ubs_feature, axis=1)
        # concat feature
        total_feature_embedding = tf.concat([
            self.flatten(target_ad),
            self.flatten(ubs_feature),
            self.flatten(profile_feature),
            context_feature
        ], axis=-1)
        # expert outs
        expert_outs = tf.concat([
            tf.expand_dims(expert(total_feature_embedding), axis=1) for expert in self.experts_layer.values()
        ], axis=1)
        # gate outs
        ctr_gate_out = tf.reduce_sum(tf.expand_dims(self.ctr_gate(total_feature_embedding), axis=-1) * expert_outs, axis=1)
        ctcvr_gate_out = tf.reduce_sum(tf.expand_dims(self.ctcvr_gate(total_feature_embedding), axis=-1) * expert_outs, axis=1)
        # ctr, ctcvr
        ctr = self.ctr_tower(ctr_gate_out)
        ctcvr = self.ctcvr_tower(ctcvr_gate_out)
        return ctr, ctcvr

'''Deep Structured Semantic Model (DSSM)'''
class DSSM(tf.keras.Model):
    def __init__(self, emb_dim, item_feature_category_list, profile_feature_category_list) -> None:
        super().__init__()
        '''
            emb_dim: embedding维度
            item_feature_category_list: 商品特征类别列表
            profile_feature_category_list: 用户特征类别列表
        '''
        # embedding layer
        self.embedding_layer = OrderedDict()
        for i in range(len(item_feature_category_list)):
            self.embedding_layer['ItemFc_%d'%(i)] = tf.keras.layers.Embedding(item_feature_category_list[i], emb_dim)
        for i in range(len(profile_feature_category_list)):
            self.embedding_layer['ProfileFc_%d'%(i)] = tf.keras.layers.Embedding(profile_feature_category_list[i], emb_dim)
        
        # tower [召回模型结果不需要sigmoid概率化，只需要ranking序正确即可]
        self.item_tower = tf.keras.Sequential([
            tf.keras.layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l1(0.001)), # 第一次l1正则筛选特征，隐层l2正则
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(emb_dim) # 输出为 item_embedding
        ])
        self.user_tower = tf.keras.Sequential([
            tf.keras.layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l1(0.001)), # 第一次l1正则筛选特征，隐层l2正则
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(emb_dim) # 输出为 user_embedding
        ])
        self.flatten = tf.keras.layers.Flatten()

    def call(self, target_ad, profile_feature):
        '''
            target_ad: 待预测商品特征张量 (sparse特征) [Batch_size, Feature_num]
            profile_feature: 用户特征张量 (sparse特征) [Batch_size, Feature_num]
        '''
        # 特征 Embedding 化 (这里的embedding是初始化embedding，不包含 user 与 item 的匹配信息)
        target_ad = tf.concat([self.embedding_layer['ItemFc_%d'%(i)](target_ad[:, i:i+1]) for i in range(target_ad.shape[-1])], axis=-2)
        profile_feature = tf.concat([self.embedding_layer['ProfileFc_%d'%(i)](profile_feature[:, i:i+1]) for i in range(profile_feature.shape[-1])], axis=-2)
        # 获取 item_embedding 和 user_embedding （这里的 user_embedding 和 item_embedding 带有 user 与 item 的匹配信息，根据 id 进行更新并保存，用于召回）
        # 通常serving时，所有users共享一个user_tower的forward，获取user_embedding，每个item_id有独立的item_tower，将item_embedding保存备用
        item_embedding = self.item_tower(self.flatten(target_ad))
        user_embedding = self.user_tower(self.flatten(profile_feature))
        # out
        out = tf.expand_dims(tf.reduce_sum(item_embedding * user_embedding, axis=-1), axis=-1)
        return out

'''YouTubeDNN'''
class YouTubeDNN(tf.keras.Model):
    def __init__(self, emb_dim, item_feature_category_list, profile_feature_category_list) -> None:
        super().__init__()
        '''
            emb_dim: embedding维度
            item_feature_category_list: 商品特征类别列表
            profile_feature_category_list: 用户特征类别列表
        '''
        # embedding layer
        self.embedding_layer = OrderedDict()
        for i in range(len(item_feature_category_list)):
            self.embedding_layer['ItemFc_%d'%(i)] = tf.keras.layers.Embedding(item_feature_category_list[i], emb_dim)
        for i in range(len(profile_feature_category_list)):
            self.embedding_layer['ProfileFc_%d'%(i)] = tf.keras.layers.Embedding(profile_feature_category_list[i], emb_dim)
        # YouTube tower [召回模型结果不需要sigmoid概率化，只需要ranking序正确即可]
        # 倒数第二层 output 为 user_embedding, 最后一层 weights 为 item_embedding
        self.tower = tf.keras.Sequential([
            tf.keras.layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l1(0.001)),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('relu') # 输出为 user_embedding
        ])
        self.item_embedding_layer = tf.keras.layers.Dense(1, use_bias=False) # weights 为 item_embedding
        self.flatten = tf.keras.layers.Flatten()

    def call(self, target_ad, profile_feature):
        '''
            target_ad: 待预测商品特征张量 (sparse特征) [Batch_size, Feature_num]
            profile_feature: 用户特征张量 (sparse特征) [Batch_size, Feature_num]
        '''
        # 特征 Embedding 化 (这里的embedding是初始化embedding，不包含 user 与 item 的匹配信息)
        target_ad = tf.concat([self.embedding_layer['ItemFc_%d'%(i)](target_ad[:, i:i+1]) for i in range(target_ad.shape[-1])], axis=-2)
        profile_feature = tf.concat([self.embedding_layer['ProfileFc_%d'%(i)](profile_feature[:, i:i+1]) for i in range(profile_feature.shape[-1])], axis=-2)
        # 获取 item_embedding 和 user_embedding （这里的 user_embedding 和 item_embedding 带有 user 与 item 的匹配信息，根据 id 进行更新并保存，用于召回）
        # 由于 weights 表示为 item_embedding，所以每个item有独立的YouTubeDNN，serving时item_embedding直接取出weights，user_embedding前向计算获取output
        total_embedding = tf.concat([
            self.flatten(target_ad),
            self.flatten(profile_feature)
        ], axis=-1)
        user_embedding = self.tower(total_embedding)
        out = self.item_embedding_layer(user_embedding)
        # item_embddding = self.item_embedding_layer.weights[0]
        return out