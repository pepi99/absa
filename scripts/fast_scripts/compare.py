import pandas as pd

def create_comp_df(df_arr, take_diff=False, take_neg=False):
    # df1 = pd.read_csv('/Users/petar.ulev/Documents/absa/data/comparison_data/15000-20000/res_preprocessed_model1_15000-20000.csv')
    # df2 = pd.read_csv('/Users/petar.ulev/Documents/absa/data/comparison_data/15000-20000/res_preprocessed_model2_15000-20000.csv')
    # df3 = pd.read_csv('/Users/petar.ulev/Documents/absa/data/comparison_data/15000-20000/res_preprocessed_model3_15000-20000.csv')
    #
    # df_arr = [df1, df2, df3]
    df1 = df_arr[0]



    df_comp = pd.DataFrame()
    df_comp['embedded_text'] = df1['embedded_text']

    for j, df in enumerate(df_arr):
        mdl_idx = f'm{j+1}'
        df_comp[f'sentiment_{mdl_idx}'] = df['sentiment']
        df_comp[f'confidence_{mdl_idx}'] = df['confidence']
        df_comp[f'coin_{mdl_idx}'] = df['coin']
    if take_diff:
        df_comp = df_comp[((df_comp['sentiment_m1'] != df_comp['sentiment_m2']) |
                           (df_comp['sentiment_m2'] != df_comp['sentiment_m3']) |
                           (df_comp['sentiment_m1'] != df_comp['sentiment_m3'])
                           )]
        df_comp.to_csv('data/comparison/20000-21000/compare_diff.csv', index=False)
        df_comp[['embedded_text']].drop_duplicates().to_csv('data/comparison/20000-21000/compare_diff_textonly.csv',
                                                            index=False, header=False)

    elif take_neg:
        df_comp = df_comp[((df_comp['sentiment_m1'] == 'Negative') |
                       (df_comp['sentiment_m2'] == 'Negative') |
                           (df_comp['sentiment_m3'] == 'Negative')
                           )]
        df_comp.to_csv('data/comparison/20000-21000/compare_neg.csv', index=False)
        df_comp[['embedded_text']].drop_duplicates().to_csv('data/comparison/20000-21000/compare_neg_textonly.csv',
                                                            index=False, header=False)
    else:
        df_comp.to_csv('/Users/petar.ulev/Documents/absa/data/comparison_data/23000-23500/compare_all.csv', index=False)
        df_comp[['embedded_text']].drop_duplicates().to_csv('/Users/petar.ulev/Documents/absa/data/comparison_data/23000-23500/compare_all_textonly.csv',
                                                            index=False, header=False)


    # df_comp = df_comp[['embedded_text']]
    # df_comp.drop_duplicates(inplace=True)


def analyse_comp_df():
    df = pd.read_csv('../../data/comparison/20000-21000/compare_diff.csv')
    print(df[df['sentiment_m2'] != df['sentiment_m3']])