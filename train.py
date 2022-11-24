import argparse
import tensorflow as tf
import numpy as np
import random
from model import * 

def parse_args():
    parser = argparse.ArgumentParser(description='Recommend System')
    parser.add_argument(
        '--model', 
        help='select model', 
        choices= single_task_model_names + multi_task_model_names + targeting_model_names
        )
    args = parser.parse_args()
    return args

'''排序：单任务'''
def single_task(model=LogisticalRegression):
    print('\n---------------------------------Single Task Train---------------------------------\n')
    print('\nmodel:【%s】\n'% (str(model).split('.')[1][:-2]))
    model = model(emb_dim, item_feature_category_list, profile_feature_category_list)
    out = model(target_ad, ubs_feature, profile_feature, context_feature)
    print('\noutput_shape: ', out.shape, '\n')

    # 定义4个输入
    target_ad_input = tf.keras.layers.Input(shape=target_ad.shape[1:])
    ubs_feature_input = tf.keras.layers.Input(shape=ubs_feature.shape[1:])
    profile_feature_input = tf.keras.layers.Input(shape=profile_feature.shape[1:])
    context_feature_input = tf.keras.layers.Input(shape=context_feature.shape[1:])
    # 定义单个输出
    output = model(target_ad_input, ubs_feature_input, profile_feature_input, context_feature_input)
    # 构建模型
    Single_model = tf.keras.Model(
        inputs=[target_ad_input, ubs_feature_input, profile_feature_input, context_feature_input], 
        outputs=output
    )
    Single_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4), # 优化器
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), # loss函数【二分类】
        learning_rate=tf.keras.optimizers.schedules.InverseTimeDecay(5e-4, epochs//3, 0.5), # 【学习率衰减】
        metrics=['accuracy'] # 【指标】
    )
    Single_model.fit(
        [target_ad, ubs_feature, profile_feature, context_feature], # 【输入】
        ctr_label, # 【标签】
        validation_split = 0.2, #【验证集比例】
        shuffle = True,
        epochs = epochs,
        batch_size = batch_size,
    )

'''排序：多任务'''
def multi_task(model=ESMM):
    print('\n---------------------------------Multi-Task Train---------------------------------\n')
    print('\nmodel:【%s】\n'% (str(model).split('.')[1][:-2]))
    model = model(emb_dim, item_feature_category_list, profile_feature_category_list)
    out = model(target_ad, ubs_feature, profile_feature, context_feature)
    
    print('\noutput_shape: [\n')
    for each in out:
        print(each.shape, '\n')
    print(']\n')

    # 定义4个输入和2个输出
    target_ad_input = tf.keras.layers.Input(shape=target_ad.shape[1:])
    ubs_feature_input = tf.keras.layers.Input(shape=ubs_feature.shape[1:])
    profile_feature_input = tf.keras.layers.Input(shape=profile_feature.shape[1:])
    context_feature_input = tf.keras.layers.Input(shape=context_feature.shape[1:])
    ctr_out, ctcvr_out = model(target_ad_input, ubs_feature_input, profile_feature_input, context_feature_input)
    ctr_out, ctcvr_out = tf.keras.Sequential([], name='CTR')(ctr_out), tf.keras.Sequential([], name='CTCVR')(ctcvr_out) # 多输出命名，方便观察loss和metrics
    # 多任务模型构建
    Multi_model = tf.keras.Model(
        inputs=[target_ad_input, ubs_feature_input, profile_feature_input, context_feature_input],
        outputs=[ctr_out, ctcvr_out]
    )
    Multi_model.compile(
        optimizer='adam',
        learning_rate=tf.keras.optimizers.schedules.InverseTimeDecay(1e-3, epochs//3, 0.5),
        loss=[tf.keras.losses.BinaryCrossentropy(from_logits=True), tf.keras.losses.BinaryCrossentropy(from_logits=True)],
        loss_weights=[0.3, 0.7], # loss权重
        metrics=['accuracy']
    )
    Multi_model.fit(
        [target_ad, ubs_feature, profile_feature, context_feature],
        [ctr_label, ctcvr_label],
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True
    )

'''召回'''
def targeting(model=DSSM):
    print('\n---------------------------------Targeting train---------------------------------\n')
    print('\nmodel:【%s】\n'% (str(model).split('.')[1][:-2]))
    model = model(emb_dim, item_feature_category_list, profile_feature_category_list)
    out = model(target_ad, profile_feature)
    print('\noutput_shape: ', out.shape, '\n')
    # 定义双输入：用户特征与商品特征
    target_ad_input = tf.keras.layers.Input(shape=target_ad.shape[1:])
    profile_feature_input = tf.keras.layers.Input(shape=profile_feature.shape[1:])
    output = model(target_ad_input, profile_feature_input)
    # 召回模型
    Target_model = tf.keras.Model(
        inputs=[target_ad_input, profile_feature_input],
        outputs=output
    )
    Target_model.compile(
        optimizer='adam',
        learning_rate=tf.keras.optimizers.schedules.InverseTimeDecay(1e-3, epochs//3, 0.5),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    Target_model.fit(
        [target_ad, profile_feature],
        [ctr_label],
        validation_split=0.2,
        shuffle=True,
        batch_size=batch_size,
        epochs=epochs
    )

if __name__ == '__main__':
    tf.enable_eager_execution()
    single_task_model_names = ['LR', 'FM', 'DNN', 'WideAndDeep', 'DeepFM', 'DCN', 'DIN', 'Transformer']
    single_task_models = [LogisticalRegression, FactorizationMachine, DeepNeuralNetwork, WideAndDeep, DeepFM, DeepCrossNetwork, DeepIntersetNetwork, Transformer]
    multi_task_model_names = ['ESMM', 'MMOE']
    multi_task_models = [ESMM, MMOE]
    targeting_model_names = ['DSSM', 'YouTubeDNN']
    targeting_models = [DSSM, YouTubeDNN]
    args = parse_args()
    
    model_dict = dict(zip(
            single_task_model_names + multi_task_model_names + targeting_model_names,
            single_task_models +  multi_task_models + targeting_models
        ))

    n = 1000 # 样本总数
    k = 100 # 用户行为序列（ubs）长度
    m2 = 20 # 目标商品特征数【与用户行为序列商品特征数一致】
    m1 = 10 # 用户画像特征数
    m3 = 30 # 上下文特征数
    emb_dim = 128 # embedding维度

    item_feature_category_list = [random.randint(2, 10) for _ in range(m2)] # 目标商品特征类别列表
    profile_feature_category_list = [random.randint(2, 10) for _ in range(m1)] # 用户画像特征类别列表
    ctr_label = np.array([random.randint(0, 1) for _ in range(n)]).reshape(-1, 1) # ctr标签：【0、1】二分类
    ctcvr_label = np.array([random.randint(0, 1) for _ in range(n)]).reshape(-1, 1) # ctcvr标签：【0、1】二分类

    ubs_feature = np.array([ [ [ random.randint(0, item_feature_category_list[k]-1) for k in range(m2)] for j in range(k)] for i in range(n)], dtype=np.int32) # 用户行为序列特征【1000，100，20】，一条序列包含100个历史商品行为
    profile_feature = np.array([ [ random.randint(0, profile_feature_category_list[j]-1) for j in range(m1)] for i in range(n)], dtype=np.int32) # 用户画像特征 【1000，10】
    context_feature = np.random.randn(n, m3).astype(np.float32) # 上下文特征【1000，30】
    target_ad = np.array([[random.randint(0, item_feature_category_list[k]-1) for k in range(m2)] for _ in range(n)], dtype=np.int32) # 目标商品/目标广告特征 【1000，20】，与用户行为序列中的某一历史商品行为一致

    ubs_feature = tf.convert_to_tensor(ubs_feature)
    profile_feature = tf.convert_to_tensor(profile_feature)
    context_feature = tf.convert_to_tensor(context_feature)
    target_ad = tf.convert_to_tensor(target_ad)
    ctr_label = tf.convert_to_tensor(ctr_label)
    ctcvr_label = tf.convert_to_tensor(ctcvr_label)

    epochs = 10 # epochs
    batch_size = 10 # batch_size 

    print('\n==========================================推荐样本构建==========================================\n')
    print('样本总数:【%d】\n用户行为序列(ubs)长度:【%d】\n目标商品特征数【与用户行为序列商品特征数一致】:【%d】\n用户画像特征数:【%d】\n上下文特征数:【%d】\n特征embedding维度:【%d】\n'%(n, k, m2, m1, m3, emb_dim))
    print('\n用户行为序列特征 user_behaviour_sequence: %s\n用户画像特征 user_profile: %s\n上下文特征 context_feature: %s\n目标商品/广告 target_ad: %s\nCTR label: %s\nCTCVR label: %s\n'%(ubs_feature.shape, profile_feature.shape, context_feature.shape, target_ad.shape, ctr_label.shape, ctcvr_label.shape))
    print('\n==========================================开始训练==========================================\n')
    
    model_name = args.model
    # 单任务排序模型
    if model_name in single_task_model_names:
        single_task(model=model_dict[model_name])
    # 多任务排序模型
    elif model_name in multi_task_model_names:
        multi_task(model=model_dict[model_name])
    # 召回模型
    elif model_name in targeting_model_names:
        targeting(model=model_dict[model_name])

    

    
