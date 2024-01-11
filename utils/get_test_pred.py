# 该文件用于为informer测试集输出的预测结果添加时间戳以及反归一化
# 该文件为补丁，独立于informer源码，跑完informer试验后，直接运行此文件即可，最终结果保存在对应的实验结果目录下。
import os
import pandas as pd
import numpy as np
from data.data_loader import Dataset_Custom

root_path="./data/"
data_path="PL.csv"
target='freq'
freq='5t'

df = pd.read_csv(f"{root_path}/{data_path}")

def get_testPred_with_time(setting, result_dir="./results"):
    # # set saved model path
    # setting = 'informer_WTH_ftM_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0'  ## 改成你自己的
    # 恢复参数
    args = setting.split("_")
    args[1:-2] = [arg[2:] for arg in args[1:-2]]
    [args_model,
    args_data, 
    args_features, 
    args_seq_len, 
    args_label_len, 
    args_pred_len, 
    args_d_model, 
    args_n_heads, 
    args_e_layers, 
    args_d_layers, 
    args_d_ff, 
    args_attn, 
    args_factor,
    args_embed, 
    args_distil, 
    args_mix, 
    args_des, 
    ii]=args

    preds = np.load(os.path.join(result_dir, setting, 'pred.npy'))
    trues = np.load(os.path.join(result_dir, setting, 'true.npy'))
    # [samples, pred_len, dimensions]
    print(preds.shape, trues.shape)

    # 恢复成序列形式
    preds = np.concatenate((preds[:, 0, :],preds[-1,1:,:]))
    trues = np.concatenate((trues[:, 0, :],trues[-1,1:,:]))
    # [pred_len, dimensions]
    print(preds.shape, trues.shape)


    # 获取原始时间戳
    test_data = Dataset_Custom(
        root_path=root_path,
        data_path=data_path,
        flag="test",
        size=[int(args_seq_len), int(args_label_len), int(args_pred_len)],
        features=args_features,
        target=target,
        inverse=False,
        timeenc=args_embed,
        freq=freq,
    )
    times = df.iloc[-len(test_data.data_x):,[0]]
    times = times[int(args_seq_len):]
    times = times[:len(trues)]

    # 反归一化
    for di in range(trues.shape[1]):
        times[f"PRED_{di}"] = 0
        times[f"TRUE_{di}"] = 0
        
    times[[f"PRED_{di}" for di in range(trues.shape[1])]] = test_data.inverse_transform(preds)
    times[[f"TRUE_{di}" for di in range(trues.shape[1])]] = test_data.inverse_transform(trues)

    # 保存成csv，剩下的就是从里面找出两列（PRED和TRUE，即预测值和真实值）来画图就行
    times.to_csv(os.path.join(result_dir, setting, "test_true_pred.csv"), index=None)


if __name__ == "__main__":
    result_dir = "./results"
    for setting_dir in os.listdir(result_dir):
        if os.path.isdir(os.path.join(result_dir, setting_dir)):
            get_testPred_with_time(setting=setting_dir, result_dir=result_dir)
