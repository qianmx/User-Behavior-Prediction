import pandas as pd
import numpy as np

# All column names are omitted here due to confidentiality

def combine_meta_data(df, meta_tuple, creative_tuple, option='both'):
    '''
    Meta data url is stored in a tuple:
    (url_meta_iso,url_meta_android,url_application)
    Creative url is stored in a tuple
    (url_creative,url_creative_tag,url_video)
    option - 'meta'/'creative'/'both', default both
    '''
    if (option == 'meta') | (option == 'both'):

        meta_iso = pd.read_csv(meta_tuple[0], sep='|',
                               quotechar='"', header=None, error_bad_lines=False, warn_bad_lines=False)
        meta_android = pd.read_csv(meta_tuple[1], sep='|',
                                   quotechar='"', header=None, error_bad_lines=False, warn_bad_lines=False)
        application = pd.read_csv(meta_tuple[2], header=None, sep='|',
                                  quotechar='"', error_bad_lines=False, warn_bad_lines=False)

        application.columns = []
        meta_iso.columns = []
        meta_android.columns = []

        meta_iso.market_id = meta_iso.market_id.astype('string')
        application.market_id = application.market_id.astype('string')
        a = pd.merge(application, meta_iso, on='market_id', how='left')
        a = pd.merge(a, meta_android, on='market_id', how='left', suffixes=['', '_android'])
        common = [i for i in a.columns if '_android' in i]
        common_iso = []
        for i in range(len(common)):
            a[common_iso[i]].fillna(a[common[i]], inplace=True)
            del a[common[i]]

        columns = a.columns.values
        columns[0] = 'advertiser_app_store_id'
        a.columns = columns

        b = pd.merge(df, a, on='advertiser_app_store_id', how='left')
        columns = a.columns.values
        columns[0] = 'publisher_app_store_id'
        a.columns = columns

        print 'start merge meta data'
        b = pd.merge(b, a, on='publisher_app_store_id', how='left', suffixes=['', '_publisher'])

        publisher_name = [i for i in b.columns if '_publisher' in i][1:]
        keep_col_name = ['_'.join(i.split('_')[0:-1]) for i in publisher_name]
        b.loc[b.is_publisher == 't', keep_col_name] = np.nan
        for i in range(len(publisher_name)):
            b[keep_col_name[i]].fillna(b[publisher_name[i]], inplace=True)
            del b[publisher_name[i]]

    if (option == 'creative') | (option == 'both'):
        if option == 'creative':
            b = df

        # creative table
        df_creative = pd.read_table(creative_tuple[0],
                                    sep='|',
                                    quotechar='"',
                                    warn_bad_lines=False,
                                    error_bad_lines=False, header=None)
        df_creative.columns = []
        # select useful cols
        creative_cols = []
        df_creative = df_creative[creative_cols]

        # creative tag table
        df_creative_tag = pd.read_table(creative_tuple[1],
                                        warn_bad_lines=False,
                                        sep='|',
                                        quotechar='"',
                                        error_bad_lines=False, header=None)
        df_creative_tag.columns = []
        df_creative_tag = df_creative_tag.groupby('creative_id').creative_tag.apply(list).reset_index()

        # vidoe table
        df_video = pd.read_table(creative_tuple[2],
                           warn_bad_lines=False,
                           sep='|',
                           quotechar='"',
                           error_bad_lines=False,header=None)
        df_video.columns = []

        print 'start merge creative data'
        b = pd.merge(b, df_creative, how='left', on='creative_id', suffixes=['', '_creative'])
        # b = pd.merge(b, df_creative_tag, how='left', on='creative_id')
        # b = pd.merge(b, df_video, how='left', on='video_id')

    return b

