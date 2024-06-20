import asyncio

from scipy.stats import pearsonr, spearmanr

from apputils.log import write_log
from config.appconfig import SEVERITY
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose


def test_correllation(df, feature_1, feature_2, ts_shift=0):
    asyncio.run(write_log(message=f'Verify :{feature_1}', severity=SEVERITY.INFO))
    pearson_corr, p_value_p = pearsonr(df[feature_1], df[feature_2] if ts_shift == 0 else df[feature_2].shift(ts_shift))
    spearmanr_corr, p_value_s = spearmanr(df[feature_1], df[feature_2])
    if p_value_p < 0.05 and p_value_s < 0.05:
        return True
    return False


def make_stationary(df, feature):
    df_regs = df['region'].unique().tolist()
    SR = len(df_regs)
    subdfs = list(map(lambda x: df.loc[df['region'] == x][['date_reg', feature]], df_regs))
    #
    reg_s = list(zip(df_regs, subdfs))
    # non_s = [x for x in reg_s if x[1] >= 0.05]
    result_df = pd.DataFrame()

    for item in reg_s:
        try:
            serie = item[1]
            serie.set_index('date_reg', inplace=True)
            decompose_result = seasonal_decompose(serie[feature].squeeze(), model="multiplicative")
            rt = pd.DataFrame(decompose_result.trend)
            rt[feature] = rt['trend'].fillna(0)
            rt.drop('trend', axis=1, inplace=True)
            rst = pd.DataFrame(decompose_result.seasonal)
            rst[feature] = rst['seasonal'].fillna(0)
            rst.drop('seasonal', axis=1, inplace=True)
            rsd = pd.DataFrame(decompose_result.resid)
            rsd[feature] = rsd['resid'].fillna(0)
            rsd.drop('resid', axis=1, inplace=True)
            statrs = serie.subtract(rt).subtract(rsd).subtract(rst)
            statrs['region'] = item[0]
            result_df = pd.concat([result_df, statrs], axis=0)
        except:
            pass
        stationaries = list(map(lambda x: adfuller(x[feature]),result_df))
    return None


def lag_detect(df, feature_1, feature_2):
    maxlen = int(len(df[feature_2]) / 2)
    for i in range(maxlen):
        if test_correllation(df, feature_1, feature_2, i):
            return i, True
        else:
            return -1, False


def apply_stationery(df, sel_field, feature):
    pass
